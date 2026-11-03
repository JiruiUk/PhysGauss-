"""Evaluation-only repair: original trajectory RGB includes held-out test frames.

Frozen training/evaluation sources, checkpoints and first failed export remain
unchanged. Redirect all repaired outputs to evaluation_v2, never overwrite.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import DATA, sha, write, read
from variants.staged_reconstruction.cross_temporal_static import evaluate as original


def main():
    report = original.REPORT
    assert not (report/'exports.json').exists(), 'Preserve completed evaluations'
    source = Path(__file__).resolve()
    fixed_source = ROOT/'variants/staged_reconstruction/cross_temporal_static/evaluate.py'
    assert sha(fixed_source) == read(report/'source_freeze.json')[fixed_source.relative_to(ROOT).as_posix()]
    original.OUT = original.OUT/'evaluation_v2'; original.OUT.mkdir(parents=True, exist_ok=False)
    ordinary_rgb = original.rgb
    fallback_rows = []
    raw_root = ROOT/'data/PIDG/rubber_duck'
    def rgb_with_original_trajectory(frame, width):
        if (DATA/(frame['file_path']+'.png')).exists(): return ordinary_rgb(frame, width)
        path = (raw_root/(frame['file_path']+'.png')).resolve()
        assert path.is_relative_to(raw_root.resolve()) and path.exists(), path
        with Image.open(path) as image:
            rgba = np.asarray(image.convert('RGBA').resize((width, width*9//16), Image.Resampling.BILINEAR), dtype=np.float32)/255
        fallback_rows.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'width': width,
                              'use': 'observed_RGB_in_original_trajectory_video_only'})
        return rgba[..., :3]*rgba[..., 3:4]
    original.rgb = rgb_with_original_trajectory
    write(report/'evaluation_failure_attempt1.json', {
        'stage': 'video_export', 'exception': 'FileNotFoundError: derived training/train/0001.png',
        'reason': 'Full200-camera trajectory includes test images absent from derived training directory',
        'failed_source_sha256': sha(fixed_source), 'failed_log': 'evaluate.log',
        'failed_outputs_preserved': 'output/staged_reconstruction/S1_cross_temporal (root files)',
        'training_and_weights_modified': False, 'repaired_source_sha256': sha(source),
        'repair_scope': 'RGB fallback from original dataset only when derived file absent; new output evaluation_v2; no training or metric threshold changes'})
    original.main()
    write(report/'evaluation_repair_integrity.json', {'passed': True, 'repair_source_sha256': sha(source),
        'frozen_evaluation_source_unchanged': sha(fixed_source), 'original_dataset_RGB_fallback': fallback_rows,
        'outputs': original.OUT.relative_to(ROOT).as_posix(), 'training_weights_and_protocol_unchanged': True})


if __name__ == '__main__': main()
