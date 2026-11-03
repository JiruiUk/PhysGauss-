"""Make browser-compatible copies; preserve immutable experiment video exports."""
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/staged_reconstruction'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    result = {'codec_tag': ''.join(chr((code >> (8 * i)) & 255) for i in range(4)),
              'fps': cap.get(cv2.CAP_PROP_FPS),
              'size': [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert list(frame.shape[:2][::-1]) == result['size']
        count += 1
    cap.release()
    result['decoded_frames'] = count
    return result


def atoms(path):
    names = []
    with path.open('rb') as stream:
        while header := stream.read(8):
            length, name = struct.unpack('>I4s', header)
            header_size = 8
            if length == 1:
                length = struct.unpack('>Q', stream.read(8))[0]
                header_size = 16
            names.append(name.decode('ascii'))
            if length == 0:
                break
            assert length >= header_size
            stream.seek(length - header_size, 1)
    return names


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    originals = json.loads((REPORT / 'completion_integrity.json').read_text())['videos_verified']
    receipt_file = REPORT / 'browser_video_conversion.json'
    prior = json.loads(receipt_file.read_text()) if receipt_file.exists() else None
    records = []
    for item in originals:
        stage = item['stage']
        folder = 'S2_geometry_recovery' if stage == 'S2' else stage
        source = ROOT / 'output/staged_reconstruction' / folder / (item['view'] + '.mp4')
        target = source.with_name(source.stem + '.browser.mp4')
        assert sha(source) == item['sha256'], 'Historical video changed'
        if target.exists():
            assert prior is not None, 'Unrecorded browser copy; preserve it for inspection'
            previous = next(x for x in prior['videos'] if x['file'] == target.relative_to(ROOT).as_posix())
            assert sha(target) == previous['sha256']
            records.append(previous)
            continue
        temporary = target.with_name(target.stem + '.partial.mp4')
        log = REPORT / f'{stage}_{item["view"]}_h264.log'
        args = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-i', str(source),
                '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '18', '-profile:v', 'main', '-level:v', '3.1',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
        with log.open('w', encoding='utf8') as output:
            subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)
        metadata = inspect(temporary)
        boxes = atoms(temporary)
        assert metadata['codec_tag'] in ['avc1', 'h264'], metadata
        assert metadata['fps'] == 20.0 and metadata['size'] == [800, 450] and metadata['decoded_frames'] == 200, metadata
        encoding_log = log.read_text(encoding='utf8')
        assert 'profile Main, level 3.1' in encoding_log and 'yuv420p' in encoding_log and 'avc1' in encoding_log
        assert boxes.index('moov') < boxes.index('mdat'), boxes
        assert sha(source) == item['sha256']
        temporary.rename(target)
        records.append({'stage': stage, 'view': item['view'], 'file': target.relative_to(ROOT).as_posix(),
                        'source_sha256': item['sha256'], 'sha256': sha(target), **metadata,
                        'profile': 'H.264 Main, level3.1, yuv420p', 'faststart': True})
        # Retain a receipt after every successful copy to allow safe recovery.
        receipt_file.write_text(json.dumps({'scope': 'Display copies only; no model, native image, metric or original video changes',
                                           'ffmpeg': ffmpeg, 'videos': records}, indent=2) + '\n', encoding='utf8')
        print('BROWSER_VIDEO_VERIFIED', stage, item['view'], flush=True)


if __name__ == '__main__':
    main()
