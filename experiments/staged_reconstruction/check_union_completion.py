"""Verify completed S3-U artifacts and preserve historical results without training."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from html.parser import HTMLParser
import msvcrt
from pathlib import Path
import struct
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import cv2
import imageio_ffmpeg
import lpips

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import REPORT, read, write, sha
from variants.staged_reconstruction.union.prepare import UNION_REPORT, UNION_RUNS, verify_sources


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in {'href', 'src', 'poster'} and v)


def exited(pid):
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        assert ctypes.get_last_error() in {87, 1168}, (pid, ctypes.get_last_error())
        return True
    try:
        code = wintypes.DWORD()
        assert api.GetExitCodeProcess(handle, ctypes.byref(code))
        assert code.value != 259, f'Known task process still active: {pid}'
        return True
    finally:
        api.CloseHandle(handle)


def check_video(row):
    path = ROOT/row['path']
    assert sha(path) == row['sha256']
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    assert frames == row['frames'] == 200 and fps == row['fps'] == 20
    assert size == ([1600, 480] if '_S3_vs_union' in path.stem else [800, 450])
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-i', str(path),
        '-map', '0:v:0', '-c', 'copy', '-frames:v', '0', '-f', 'null', '-'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    assert 'Video: h264 (Main)' in probe.stderr and 'yuv420p' in probe.stderr
    atoms = {}
    with path.open('rb') as stream:
        while stream.tell() < path.stat().st_size:
            offset = stream.tell()
            length, kind = struct.unpack('>I4s', stream.read(8))
            if length == 1:
                length = struct.unpack('>Q', stream.read(8))[0]
            elif length == 0:
                length = path.stat().st_size-offset
            assert length >= 8
            atoms[kind] = offset
            stream.seek(offset+length)
    assert atoms[b'moov'] < atoms[b'mdat']
    return {**row, 'display_size': size, 'metadata_probe_passed': True,
        'moov_before_mdat': True, 'full_decode_200_frames_receipt': 'physics_exports.json'}


def main():
    verify_sources()
    final = read(UNION_REPORT/'final_summary.json')
    assert final['training_completed'] and final['preselected_final_iteration'] == 6000
    checkpoints = []
    for phase, updates in [('warmup', 16000), ('physics', 6000)]:
        directory = UNION_RUNS/phase/'formal'
        manifest = read(directory/'checkpoint.json')
        audit = read(UNION_REPORT/f'{phase}_state_audit.json')
        actual = sha(directory/'checkpoint.pt')
        assert actual == manifest['sha256'] == audit['checkpoint_sha256']
        assert manifest['iteration'] == audit['iteration'] == updates
        assert all(audit['numeric_gates'].values())
        assert audit['static_exact_S1'] and audit['all_actual_Adam_counters_correct']
        assert audit['exact_all_step_sampling_and_final_python_RNG']
        assert audit['numpy_CPU_Torch_all_CUDA_RNG_match_no_random_forward_contract']
        assert read(UNION_REPORT/f'{phase}_smoke_verification.json')['passed']
        checkpoints.append({'phase': phase, 'updates': updates, 'sha256': actual,
            'path': (directory/'checkpoint.pt').relative_to(ROOT).as_posix()})
    assert read(UNION_REPORT/'test_receipt.json')['passed'] == 49
    assert read(UNION_REPORT/'native_preflight.json')['passed']
    historical = read(REPORT/'completion_integrity.json')
    for row in historical['final_checkpoints']:
        assert sha(ROOT/row['path']) == row['sha256']
    c1 = read(REPORT/'C1_baseline_comparison.json')['protected_checkpoints']
    for path, expected in c1.items():
        assert sha(ROOT/path) == expected
    for row in historical['videos_verified']:
        folder = 'S2_geometry_recovery' if row['stage'] == 'S2' else row['stage']
        assert sha(ROOT/f'output/staged_reconstruction/{folder}/{row["view"]}.mp4') == row['sha256']
    exports = read(UNION_REPORT/'physics_exports.json')
    manual = read(UNION_REPORT/'physics_manual_review.json')
    assert manual['checkpoint_sha256'] == exports['checkpoint_sha256'] == checkpoints[1]['sha256']
    for path, expected in {**exports['images'], **manual['viewed_image_sha256']}.items():
        assert sha(ROOT/path) == expected
    assert exports['camera_matrices_unchanged_after_render']
    videos = [check_video(row) for row in exports['videos']]
    pages = [ROOT/'output/staged_reconstruction/index.html',
        ROOT/'output/staged_reconstruction/S3_union/physics/index.html',
        ROOT/'output/staged_reconstruction/baseline_comparison/index.html']
    checked_links = []
    for page in pages:
        parser = Links()
        parser.feed(page.read_text(encoding='utf8'))
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (page.parent/unquote(url.path)).resolve()
            assert target.is_relative_to(ROOT) and target.is_file(), (page, link)
            checked_links.append({'page': page.relative_to(ROOT).as_posix(), 'link': link})
    contract = read(REPORT/'S3/native_full_quality.json')['contract']
    weights = {ROOT/'reproduction/torch-cache/hub/checkpoints/vgg16-397923af.pth': contract['VGG_sha256'],
        Path(lpips.__file__).parent/'weights/v0.1/vgg.pth': contract['LPIPS_linear_sha256']}
    for path, expected in weights.items():
        assert sha(path) == expected
    processes = [read(UNION_REPORT/f'{phase}_launch.json')['pid'] for phase in ['warmup', 'physics']]
    processes.append(266940)  # Recorded final review/export process; no process is started here.
    assert all(exited(pid) for pid in processes)
    with (UNION_RUNS/'training.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    write(UNION_REPORT/'completion_integrity.json', {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_source_input_protocol_files_verified': len(read(UNION_REPORT/'source_freeze.json')),
        'all_frozen_sources_inputs_protocols_match': True, 'checkpoints': checkpoints,
        'historical_final_checkpoints_preserved': historical['final_checkpoints'],
        'C1_comparison_protected_checkpoints_preserved': c1,
        'historical_original_videos_preserved': historical['videos_verified'],
        'manual_images_hash_verified': len(manual['viewed_image_sha256']),
        'export_images_hash_verified': len(exports['images']), 'videos': videos,
        'fixed_camera_fingerprints': exports['fixed_camera_fingerprints'],
        'camera_matrices_unchanged_after_render': True, 'local_HTML_links_verified': checked_links,
        'LPIPS_weights_match_original_S3_contract': {str(k): v for k, v in weights.items()},
        'known_task_processes_exited': processes, 'training_lock_free': True,
        'all_new_training_budgets_completed': True, 'training_updates': 22000,
        'geometry_or_physical_inference_success': False, 'browser_playback_observed': False,
        'no_automatic_follow_on_training': True,
        'preserved_unrelated_tracked_changes': ['experiments/integrity_feedback/proposals.json',
            'experiments/results.json', 'experiments/results.md']})
    write(UNION_RUNS/'state.json', {'phase': 'physics', 'status': 'completed_reviewed', 'iteration': 6000,
        'warmup_updates': 16000, 'total_updates': 22000, 'physical_body_number': None,
        'report': 'experiments/staged_reconstruction/S3_union/assessment.md',
        'completion_integrity': 'experiments/staged_reconstruction/S3_union/completion_integrity.json',
        'no_automatic_follow_on_training': True})
    print('S3_UNION_COMPLETION_VERIFIED', len(checkpoints), 'checkpoints;', len(videos), 'new videos;',
        len(checked_links), 'local HTML links; historical results preserved', flush=True)


if __name__ == '__main__':
    main()
