"""Independent post-training checks; never launches training or changes selection."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from output.c1_flow_ablation.reuse_control import CONTROL, TREATMENT, QUEUE, REPORTS, OUT, verify_control
from variants.c1_flow_ablation.run import assert_frozen


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    assert_frozen(ROOT/'variants/c1_flow_ablation/code')
    plan = verify_control()
    lock = read(QUEUE/'selection_lock.json')
    proof = read(REPORTS/'final_verification_reuse.json')
    assert sha(QUEUE/'selection_lock.json') == proof['selection_lock_sha256']
    frame_ids = [int(Path(f['file_path']).stem) for f in read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames']]
    assert frame_ids[21] == 85
    checked = {}
    for directory in [CONTROL, TREATMENT]:
        name = directory.parent.name
        assert lock['models'][name] == read(directory/'metric_best.json')['PSNR']
        receipt = read(directory/'checkpoints/resume.json')
        assert receipt['iteration'] == 50000 and receipt['sha256'] == sha(directory/'checkpoints/resume.pt')
        assert receipt['sha256'] == proof['models'][name]['full_checkpoint_sha256']
        for relative, expected in proof['models'][name]['snapshot_sha256'].items():
            assert sha(directory/relative) == expected
        per_view = read(directory/'per_view.json')
        max_error = 0.
        videos = []
        for tag in ['best', '50000']:
            for i in range(50):
                filename = f'{i:05d}.png'
                source = directory/'test'/('ours_'+tag)
                with Image.open(source/'gt'/filename) as gt, Image.open(source/'renders'/filename) as render, Image.open(OUT/name/'pairs'/tag/filename) as pair:
                    gt, render, pair = gt.convert('RGB'), render.convert('RGB'), pair.convert('RGB')
                    assert gt.size == render.size == (1600, 900) and pair.size == (3200, 950)
                    assert pair.crop((0,50,1600,950)).tobytes() == gt.tobytes()
                    assert pair.crop((1600,50,3200,950)).tobytes() == render.tobytes()
                    mse = np.mean((np.asarray(gt, dtype=np.float64)-np.asarray(render, dtype=np.float64))**2)/255**2
                    error = abs(-10*math.log10(max(mse,1e-12))-per_view['ours_'+tag]['PSNR'][filename])
                    assert error < 1e-4
                    max_error = max(max_error,error)
            movie = OUT/name/('GT_Render_'+tag+'.mp4')
            # Decode all video frames, including final snapshots, rather than trusting metadata.
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(movie),'-vf','scale=320:95','-pix_fmt','rgb24','-f','rawvideo','-'],capture_output=True,check=True)
            assert len(result.stdout) == 50*320*95*3
            videos.append({'file':str(movie.relative_to(ROOT)),'decoded_frames':50,'sha256':sha(movie)})
        archive = OUT/(name+'_GT_Render.zip')
        assert sha(archive) == proof['models'][name]['zip_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == {p.relative_to(OUT/name).as_posix() for p in (OUT/name).rglob('*') if p.is_file() and not p.name.endswith('.encode.log')}
            for entry in z.namelist():
                assert hashlib.sha256(z.read(entry)).hexdigest() == sha(OUT/name/entry)
        checked[name] = {'pairs':100,'independent_PSNR_max_difference':max_error,'videos':videos,'zip_CRC_and_all_member_bytes':True}
    # Preserve all existing choices and immutable original bytes through verification.
    verify_control()
    result = {'status':'passed','source_data_SHA_verified':True,'common_3k_sha256':plan['common_3k_sha256'],
              'historical_control_unchanged':True,'selection_lock_sha256':sha(QUEUE/'selection_lock.json'),
              'original_frame_85_pair_index':21,'models':checked}
    (REPORTS/'independent_verification_reuse.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
