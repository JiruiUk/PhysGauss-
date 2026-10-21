"""Evaluate retained/missing frames at frozen PIDG best snapshots; never train.

Only newly rendered retained views need new metric inference. Existing official
and missing-view per-image metrics are validated and reused. Means follow PIDG:
arithmetic mean of per-image scores, including per-image PSNR in dB.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent))
from pidg_benchmark import METRICS, read_json, runtime_hashes, sha256, write_json, write_text_atomic
from prepare import SOURCE, link_directory, save_frozen_json, fixture_hashes

REPO = ROOT / 'third_party/PIDG'
REPORTS = ROOT / 'experiments/random_frame_mask'
CONFIG = REPO / 'arguments/pidg/rubber_dack.py'


def validate_partition(original, official, retained, removed):
    """Reject overlap, incomplete coverage, altered poses/times and duplicate IDs."""
    def indexed(frames):
        result = {f['file_path']: f for f in frames}
        if len(result) != len(frames):
            raise ValueError('Duplicate source frame ID')
        return result
    full, test, kept, dropped = map(indexed, (original, official, retained, removed))
    if set(full) & set(test):
        raise ValueError('Original train/test overlap')
    if set(kept) & set(dropped):
        raise ValueError('Retained/removed overlap')
    if set(kept) | set(dropped) != set(full):
        raise ValueError('Incomplete original training-frame coverage')
    if any(full[k] != v for part in (kept, dropped) for k, v in part.items()):
        raise ValueError('Source frame metadata changed')


def mean_metrics(rows):
    if not rows:
        return None
    ids = [r['source_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evaluation frame')
    for row in rows:
        if not all(math.isfinite(row['metrics'][key]) for key in METRICS):
            raise ValueError('Nonfinite metric')
    return {key: math.fsum(row['metrics'][key] for row in rows) / len(rows) for key in METRICS}


def frame_names(count):
    return {f'{i:05d}.png' for i in range(count)}


def validate_images(directory, count):
    for kind in ('renders', 'gt'):
        if {p.name for p in (directory / kind).glob('*.png')} != frame_names(count):
            raise ValueError('Incomplete/extra PNG images: ' + str(directory / kind))


def existing_views(run, frames):
    directory = run / 'test/ours_best'
    validate_images(directory, len(frames))
    per = read_json(run / 'per_view.json')['ours_best']
    for key in METRICS:
        if set(per[key]) != frame_names(len(frames)):
            raise ValueError('Existing per-image metrics have incomplete coverage')
    rows = [{'source_id': f['file_path'], 'time': f['time'], 'png': f'{i:05d}.png',
             'render': str(directory / 'renders' / f'{i:05d}.png'),
             'gt': str(directory / 'gt' / f'{i:05d}.png'),
             'metrics': {key: per[key][f'{i:05d}.png'] for key in METRICS}}
            for i, f in enumerate(frames)]
    summary = read_json(run / 'results.json')['ours_best']
    means = mean_metrics(rows)
    if any(abs(means[key] - summary[key]) > 1e-5 for key in METRICS):
        raise ValueError('Existing per-image metrics do not reproduce saved summary')
    return rows


def frozen_inputs(run):
    paths = [run / name for name in ('cfg_args', 'reproduction.json', 'results.json', 'per_view.json')]
    for folder in ('point_cloud/iteration_best', 'deform/iteration_best'):
        paths.extend(p for p in (run / folder).rglob('*') if p.is_file())
    if (run / 'masked_holdout/results.json').exists():
        paths += [run / 'masked_holdout/results.json', run / 'masked_holdout/per_view.json']
    return {str(p): sha256(p) for p in paths}


def retained_dataset(percent, train, kept):
    target = ROOT / f'data/pidg_random_frame_mask_evaluation/drop{percent:02d}_seed0/retained_dataset'
    target.mkdir(parents=True, exist_ok=True)
    for child in SOURCE.iterdir():
        if child.is_dir():
            link_directory(target / child.name, child)
    points = target / 'points3d.ply'
    if not points.exists():
        shutil.copyfile(SOURCE / 'points3d.ply', points)
    if sha256(points) != sha256(SOURCE / 'points3d.ply'):
        raise ValueError('Evaluation canonical points changed')
    save_frozen_json(target / 'transforms_train.json', {**deepcopy(train), 'frames': deepcopy(kept)})
    # PIDG keys camera metadata by file_path across both splits. Reusing an ID
    # overwrites training flow metadata with test metadata. An image-directory
    # alias gives test cameras distinct loader IDs but exactly the same PNGs,
    # poses and times; reporting continues to use original source IDs.
    test_frames = retained_test_frames(kept)
    link_directory(target / 'retained_eval_images', SOURCE / 'train')
    save_frozen_json(target / 'transforms_test.json', {**deepcopy(train), 'frames': test_frames})
    return target


def retained_test_frames(frames):
    result = deepcopy(frames)
    for frame in result:
        original = Path(frame['file_path'])
        if original.parent.as_posix() != 'train':
            raise ValueError('Expected original train image paths')
        frame['file_path'] = 'retained_eval_images/' + original.name
    return result


def render_retained(run, dataset, frames, env):
    output = run / 'retained_evaluation'
    output.mkdir(exist_ok=True)
    for name in ('point_cloud', 'deform'):
        link_directory(output / name, run / name)
    cfg = output / 'cfg_args'
    if not cfg.exists():
        shutil.copyfile(run / 'cfg_args', cfg)
    if sha256(cfg) != sha256(run / 'cfg_args'):
        raise ValueError('Evaluation model config changed')
    expected = {'source': str(dataset), 'run': str(run), 'iteration': 'best', 'resolution_width': 1600,
                'source_ids': [f['file_path'] for f in frames],
                'transforms_sha256': sha256(dataset / 'transforms_test.json'),
                'frozen_inputs': frozen_inputs(run), 'runtime': runtime_hashes(REPO)}
    save_frozen_json(output / 'protocol.json', expected)
    marker = output / 'render_complete.json'
    raw = output / 'test/ours_best'
    if marker.exists():
        validate_images(raw, len(frames))
        return output
    command = [sys.executable, '-u', str(ROOT / 'scripts/pidg_variant_entrypoint.py'), '--repo', str(REPO),
               '--seed', '0', 'render.py', '-s', str(dataset), '-m', str(output), '--conf', str(CONFIG),
               '--load2gpu_on_the_fly', '--resolution', '1600', '--iteration', 'best', '--skip_train']
    write_json(output / 'render_command.json', command)
    with (output / 'render.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Render failed ({result.returncode}): {output / "render.log"}')
    validate_images(raw, len(frames))
    write_json(marker, {'frames': len(frames), 'completed_utc': datetime.now(timezone.utc).isoformat()})
    return output


class MetricEvaluator:
    def __init__(self):
        # Exact functions and LPIPS options from the official metrics.py, streamed
        # one PNG pair at a time to avoid loading an entire split onto the GPU.
        sys.path.insert(0, str(REPO))
        import torch
        import lpips
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
        from pytorch_msssim import ms_ssim
        self.torch, self.ssim, self.psnr, self.ms_ssim = torch, ssim, psnr, ms_ssim
        self.lpips = lpips.LPIPS(net='vgg').cuda().eval()

    def evaluate(self, render, gt):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        with Image.open(render) as image:
            if image.size != (1600, 900):
                raise ValueError('Unexpected render resolution')
            a = to_tensor(image).unsqueeze(0)[:, :3].cuda()
        with Image.open(gt) as image:
            if image.size != (1600, 900):
                raise ValueError('Unexpected GT resolution')
            b = to_tensor(image).unsqueeze(0)[:, :3].cuda()
        with self.torch.no_grad():
            return {'PSNR': self.psnr(a, b).item(), 'SSIM': self.ssim(a, b).item(),
                    'LPIPS': self.lpips(a, b).item(),
                    'MS-SSIM': self.ms_ssim(a, b, data_range=1, size_average=True).item()}


def evaluate_retained(output, frames, evaluator):
    raw = output / 'test/ours_best'
    cache = output / 'frame_metrics.json'
    saved = read_json(cache) if cache.exists() else {'rows': []}
    rows = saved['rows']
    if len(rows) > len(frames):
        raise ValueError('Unexpected cached evaluation frames')
    for i, row in enumerate(rows):
        if row['source_id'] != frames[i]['file_path']:
            raise ValueError('Cached metric frame order mismatch')
        for kind in ('render', 'gt'):
            if sha256(row[kind]) != row[kind + '_sha256']:
                raise ValueError('Cached metric image changed')
    for i in range(len(rows), len(frames)):
        name = f'{i:05d}.png'
        render, gt = raw / 'renders' / name, raw / 'gt' / name
        row = {'source_id': frames[i]['file_path'], 'time': frames[i]['time'], 'png': name,
               'render': str(render), 'gt': str(gt), 'render_sha256': sha256(render), 'gt_sha256': sha256(gt),
               'metrics': evaluator.evaluate(render, gt)}
        rows.append(row)
        write_json(cache, {'rows': rows})
        if (i + 1) % 15 == 0 or i + 1 == len(frames):
            print(f'METRICS {output.parent.parent.name}: {i + 1}/{len(frames)}', flush=True)
    write_json(output / 'results.json', {'ours_best': mean_metrics(rows)})
    write_json(output / 'per_view.json', {'ours_best': {key: {r['png']: r['metrics'][key] for r in rows} for key in METRICS}})
    return rows


def save_reports(result):
    write_json(REPORTS / 'split_metrics.json', result)
    lines = ['# 缺帧实验：保留帧、被移除帧与全量指标', '',
             '原报告主 PSNR 仅为固定 50 帧官方测试集均值，不是整个数据集。',
             '本次使用相同 best 快照、原始 baseline、1600×900 PNG，补算保留训练帧；不重新训练或选择快照。',
             'PSNR、SSIM、LPIPS、MS-SSIM 均沿用 PIDG 的逐图指标算术平均。',
             '原训练集 150 帧 = 保留帧 + 被移除帧；全量 200 帧 = 原训练集 + 官方测试 50 帧。',
             '全量含已见训练帧，仅用于重建汇总；官方测试指标仍单列作为泛化比较。',
             'PSNR 是逐图 dB 均值，不是对整组像素先算 MSE 再转 dB。0% 的缺失帧指标不适用。', '',
             '| 缺帧 | 子集 | 帧数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MS-SSIM ↑ |',
             '|---:|---|---:|---:|---:|---:|---:|']
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(['drop_percent', 'split', 'count', *METRICS])
    names = {'retained': '保留训练帧（已见）', 'removed': '被移除训练帧（未见）',
             'original_train': '原训练集全量', 'official_test': '官方测试（固定）', 'all_200': '完整数据集'}
    for row in result['rows']:
        for key, title in names.items():
            item = row['splits'][key]
            values = item['metrics']
            cells = [f'{row["drop_percent"]}%', title, str(item['count']),
                     *(['N/A'] * len(METRICS) if values is None else [f'{values[k]:.6f}' for k in METRICS])]
            lines.append('| ' + ' | '.join(cells) + ' |')
            writer.writerow([row['drop_percent'], key, item['count'],
                             *([''] * len(METRICS) if values is None else [values[k] for k in METRICS])])
    lines += ['', '合并均值按真实帧数加权，从逐帧记录直接汇总；已检查子集不重叠且覆盖原始训练清单。',
              '原始训练帧、相机、时间、点云和模型均保留。各比例的缺失集合不同，固定官方测试才是相同视图比较。',
              '', '## 新增保留帧 render 的绝对地址', '']
    for row in result['rows']:
        lines.append(f'- {row["drop_percent"]}%：`{row["retained_render_absolute"]}`')
    lines += ['', '## 执行状态', '', result['status'], '', 'Updated UTC: ' + result['updated_utc']]
    write_text_atomic(REPORTS / 'split_metrics.md', '\n'.join(lines) + '\n')
    write_text_atomic(REPORTS / 'split_metrics.csv', buffer.getvalue())
    if result['status'] == 'completed':
        index = REPORTS / 'results.md'
        marker = '<!-- split-evaluation -->'
        original = index.read_text(encoding='utf-8').split(marker)[0].rstrip()
        extra = [marker, '## 补充：保留、缺失与全量指标', '',
                 '上表的主 PSNR 仅覆盖官方测试 50 帧。补充评估保留相同 best 快照，不重新训练。', '',
                 '| 缺帧 | 保留 PSNR | 缺失 PSNR | 原训练150 PSNR | 官方测试50 PSNR | 全部200 PSNR |',
                 '|---:|---:|---:|---:|---:|---:|']
        for row in result['rows']:
            values = [row['splits'][key]['metrics'] for key in ('retained', 'removed', 'original_train', 'official_test', 'all_200')]
            extra.append('| ' + ' | '.join([f"{row['drop_percent']}%", *['N/A' if v is None else f"{v['PSNR']:.6f}" for v in values]]) + ' |')
        extra += ['', '各项均为逐图指标的算术平均，合并按真实帧数加权；全部200含已见训练图像。',
                  '[全部四项指标、范围说明与绝对 render 地址](split_metrics.md) · [CSV](split_metrics.csv) · [JSON](split_metrics.json)',
                  '', '[更新后的 PDF 报告](../../output/pdf/PIDG_MLP_MaskedData_Report.pdf)']
        write_text_atomic(index, original + '\n\n' + '\n'.join(extra) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratios', type=int, nargs='+', default=[0, 10, 30, 50, 70, 90])
    args = parser.parse_args()
    original = read_json(SOURCE / 'transforms_train.json')
    official = read_json(SOURCE / 'transforms_test.json')
    snapshot = read_json(REPORTS / 'results.json')
    before = {'runtime': runtime_hashes(REPO), 'original_dataset': fixture_hashes(SOURCE)}
    env = os.environ.copy()
    env.pop('PIDG_CONSTITUTIVE_CONFIG', None)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PIDG_HASH_SOURCE_DIR'] = str(REPO / 'hashencoder')
    env['PIDG_HASH_BUILD_DIR'] = str(REPO / 'tmp_build')
    result = {'status': 'evaluating', 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': 'frozen best snapshot; original per-image metrics; no training',
              'aggregation': 'arithmetic per-image means; unions are frame-count weighted',
              'inputs': before, 'rows': []}
    evaluator = None
    try:
        for percent in args.ratios:
            source = next(r for r in snapshot['rows'] if r['drop_percent'] == percent)
            run = Path(source['run'])
            info = read_json(run / 'reproduction.json')
            if info['status'] != 'completed' or (info['iterations'], info['resolution_width'], info['seed']) != (50000, 1600, 0):
                raise ValueError('Incomplete or mismatched source run')
            frozen = frozen_inputs(run)
            kept_doc = original if percent == 0 else read_json(ROOT / f'data/pidg_random_frame_mask/rubber_duck/drop{percent:02d}_seed0/dataset/transforms_train.json')
            missing_doc = {'frames': []} if percent == 0 else read_json(ROOT / f'data/pidg_random_frame_mask/rubber_duck/drop{percent:02d}_seed0/heldout_dataset/transforms_test.json')
            kept, missing = kept_doc['frames'], missing_doc['frames']
            validate_partition(original['frames'], official['frames'], kept, missing)
            print(f'RENDER retained drop{percent:02d}: {len(kept)} frames', flush=True)
            result['current_drop_percent'] = percent
            result['updated_utc'] = datetime.now(timezone.utc).isoformat()
            save_reports(result)
            dataset = retained_dataset(percent, original, kept)
            output = render_retained(run, dataset, kept, env)
            if evaluator is None:
                evaluator = MetricEvaluator()
                # Parity check on an unchanged PNG pair against the saved official evaluator.
                original_values = existing_views(run, official['frames'])
                first = original_values[0]
                actual = evaluator.evaluate(first['render'], first['gt'])
                delta = {k: actual[k] - first['metrics'][k] for k in METRICS}
                if any(abs(v) > 2e-5 for v in delta.values()):
                    raise ValueError('Metric implementation parity failed: ' + str(delta))
                result['official_metric_parity_delta'] = delta
            retained_rows = evaluate_retained(output, kept, evaluator)
            removed_rows = existing_views(run / 'masked_holdout', missing) if missing else []
            official_rows = existing_views(run, official['frames'])
            train_rows = retained_rows + removed_rows
            all_rows = train_rows + official_rows
            splits = {key: {'count': len(rows), 'metrics': mean_metrics(rows)} for key, rows in
                      [('retained', retained_rows), ('removed', removed_rows), ('original_train', train_rows),
                       ('official_test', official_rows), ('all_200', all_rows)]}
            # Preserve a complete, source-ID-indexed audit trail, sorted in original order.
            id_order = {f['file_path']: i for i, f in enumerate(original['frames'] + official['frames'])}
            for rows, role in [(retained_rows, 'retained'), (removed_rows, 'removed'), (official_rows, 'official_test')]:
                for view in rows:
                    view['split'] = role
            write_json(output / 'all_frame_metrics.json', {'rows': sorted(all_rows, key=lambda r: id_order[r['source_id']])})
            best = int(re.findall(r'\d+', (run / 'deform/iteration_best/iter.txt').read_text())[-1])
            result['rows'].append({'drop_percent': percent, 'best_iteration': best, 'run': str(run),
                                   'retained_render_absolute': str(output / 'test/ours_best/renders'),
                                   'retained_gt_absolute': str(output / 'test/ours_best/gt'),
                                   'all_frame_metrics': str(output / 'all_frame_metrics.json'), 'splits': splits,
                                   'source_files_sha256': frozen})
            if frozen_inputs(run) != frozen:
                raise RuntimeError('Original run inputs changed during evaluation')
            result['updated_utc'] = datetime.now(timezone.utc).isoformat()
            save_reports(result)
            print(f'COMPLETE drop{percent:02d}: ' + str(splits), flush=True)
        if before != {'runtime': runtime_hashes(REPO), 'original_dataset': fixture_hashes(SOURCE)}:
            raise RuntimeError('Original code/dataset changed during evaluation')
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        result['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save_reports(result)


if __name__ == '__main__':
    main()
