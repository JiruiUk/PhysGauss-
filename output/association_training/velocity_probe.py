"""Read-only full-state velocity/trajectory and observed-flow comparison."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'variants/association_training/code'))
from variants.association_training.evaluate import load
from variants.metric_tuning.runtime import read_json, write_json, sha256
from output.surface_support.core import components

REPORT = ROOT / 'experiments/association_training'
MODELS = {'C1_original_50k': ROOT / 'runs/pidg_color_repair/C1_repaired/rubber_duck',
          'T1_association_52k': ROOT / 'runs/pidg_association_training/pilot/T1_association/rubber_duck',
          'O1_flow_on_50k': ROOT / 'runs/pidg_c1_flow_ablation/O1_C1_flow_on/rubber_duck'}


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT / 'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT / 'third_party/PIDG/tmp_build')
    # Load association classes before importing the historical standalone helper.
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    helper = ROOT / 'output/flow_comparison/evaluate_motion_v2.py'
    spec = importlib.util.spec_from_file_location('read_only_motion_helper', helper)
    motion = importlib.util.module_from_spec(spec); spec.loader.exec_module(motion)
    manifest = read_json(motion.CACHE / 'manifest.json')
    assert len(manifest['pairs']) == 30
    assert all(sha256(motion.CACHE / p) == digest for p, digest in manifest['output_sha256'].items())
    protected = {str(d / 'checkpoints/resume.pt'): sha256(d / 'checkpoints/resume.pt') for d in MODELS.values()}
    inputs = [Path(__file__), helper, ROOT/'variants/association_training/evaluate.py',
              ROOT/'variants/kinematic_tuning/model.py', ROOT/'variants/flow_comparison/temporal.py',
              ROOT/'variants/unified_flow/flow.py', motion.CACHE/'manifest.json',
              motion.TRAIN/'transforms_train.json', motion.TRAIN/'transforms_test.json']
    import inspect
    inputs += [Path(inspect.getfile(cls)) for cls in [GaussianModel, DeformModel, motion.render]]
    for d in MODELS.values(): inputs += [d/'model_config.py', d/'metric_config.json']
    all_results = {}
    for name, directory in MODELS.items():
        os.environ['PIDG_METRIC_CONFIG'] = str(directory / 'metric_config.json')
        g, d, checkpoint = load(directory)
        ids = torch.arange(256, device='cuda') * len(g.get_xyz) // 256
        x = g.get_xyz.detach()[ids]; particle_ids = g.particle_indices[ids]
        pairs_v = []; pairs_d = []; time_rows = []; sensitivity = []; stencil_samples = []
        for t in [1/199, 51/199, 101/199, 151/199, 1.]:
            position = lambda s: motion.temporal.position(d, x, s)
            derived = motion.temporal.derivative(position, t, .001)
            predicted = motion.temporal.velocity(g, d, x, particle_ids, t, 'independent')
            error = (predicted - derived).square().mean().sqrt()
            speed = derived.square().mean().sqrt()
            time_rows.append({'time': t, 'trajectory_component_RMS': float(speed),
                              'network_component_RMS': float(predicted.square().mean().sqrt()),
                              'component_RMSE': float(error), 'relative_RMSE': float(error/speed.clamp_min(1e-8))})
            pairs_v.append(predicted.cpu().numpy()); pairs_d.append(derived.cpu().numpy())
            alternatives = []
            for h in [.0005, .002]:
                other = motion.temporal.derivative(position, t, h)
                alternatives.append(other.cpu().numpy())
                sensitivity.append(float((other-derived).square().mean().sqrt()/speed.clamp_min(1e-8)))
            stencil_samples.append(alternatives)
        v = np.stack(pairs_v); w = np.stack(pairs_d)
        vr, wr, er = [float(np.sqrt(np.mean(a.astype(np.float64)**2))) for a in [v, w, v-w]]
        denominator = np.linalg.norm(v, axis=-1) * np.linalg.norm(w, axis=-1)
        nonzero = denominator > 1e-10
        consistency = {'pooled_component_RMS_network': vr, 'pooled_component_RMS_trajectory': wr,
                       'pooled_component_RMSE': er, 'relative_RMSE': er / wr,
                       'network_to_trajectory_RMS_ratio': vr / wr,
                       'mean_direction_cosine': float((np.sum(v*w, axis=-1)[nonzero]/denominator[nonzero]).mean()),
                       'max_half_double_step_relative_change': max(sensitivity), 'times': time_rows}
        observations = []
        for pair in manifest['pairs']:
            a, b, dt = pair['source'], pair['target'], pair['dt']; t = (a-1)/199
            cam = motion.camera(motion.VAL[a]); target_cam = motion.camera(motion.FRAMES[b])
            now = motion.temporal.warp_at(d, g.get_xyz.detach(), t)
            package = motion.render(cam, g, motion.PIPE, motion.BACKGROUND,
                now['d_xyz'], now['d_rotation'], now['d_scaling'], False, override_color=torch.zeros_like(g.get_xyz))
            later = motion.temporal.warp_at(d, g.get_xyz.detach(), (b-1)/199)
            geometric, seen = motion.pixel_flow(package, *motion.next_projection(g, later, target_cam), g.get_opacity)
            velocity = motion.temporal.velocity(g, d, g.get_xyz.detach(), g.particle_indices, t, 'independent')
            current = motion.center_trajectory(g.get_xyz, now['d_xyz'])
            advected, _ = motion.pixel_flow(package, *motion.next_projection(g, later, target_cam, current+dt*velocity), g.get_opacity)
            static, _ = motion.pixel_flow(package, *motion.next_projection(g, now, target_cam, current), g.get_opacity)
            with np.load(motion.CACHE/pair['file']) as z:
                observed = torch.tensor(z['observed'], device='cuda'); valid = torch.tensor(z['valid'], device='cuda')
            image_path = motion.TRAIN / (motion.VAL[a]['file_path']+'.png')
            inputs.append(image_path)
            with Image.open(image_path) as image:
                duck = torch.tensor(components(np.array(image.convert('RGBA').resize((400,225), Image.Resampling.BILINEAR))) >= 0, device='cuda')
            row = {'source': a, 'target': b, 'model_coverage': float(seen[valid].float().mean())}
            for region, mask in [('foreground', valid), ('duck', valid & duck)]:
                assert mask.sum() > 0
                row[region+'_pixels'] = int(mask.sum())
                for key, prediction in [('trajectory', geometric), ('velocity', advected), ('static', static)]:
                    assert prediction.isfinite().all()
                    row[region+'_'+key+'_EPE_px'] = float((prediction-observed)[mask].norm(dim=-1).mean())
            observations.append(row)
        means = {k: float(np.mean([r[k] for r in observations])) for k in observations[0] if k not in ['source','target']}
        samples = ROOT/'output/association_training/velocity_samples'/f'{name}.npz'
        samples.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(samples, velocity=v, trajectory_velocity=w, alternative_stencils=np.asarray(stencil_samples),
                            kernel_ids=ids.cpu().numpy(), canonical=x.cpu().numpy())
        all_results[name] = {'checkpoint': checkpoint, 'internal_consistency': consistency,
                             'samples_path': str(samples), 'samples_sha256': sha256(samples),
                             'observed_motion_mean_over_30_pairs': means, 'observed_motion_per_pair': observations}
        print(name, json.dumps({'internal': {k:v for k,v in consistency.items() if k != 'times'}, 'observed': means}), flush=True)
        del g, d; torch.cuda.empty_cache()
    assert all(sha256(Path(p)) == digest for p, digest in protected.items())
    result = {'models': all_results, 'input_sha256': {str(p):sha256(p) for p in set(inputs)},
              'protected_checkpoints_unchanged': protected, 'optimizer_updates': 0,
              'internal_protocol': 'Full final checkpoints; same normalized0-1 time;256 evenly spaced rows x5 times; pooled component RMS; complete center includes rotation+translation; FD h=.001.',
              'observation_protocol': '30 existing validation-source to next-training frames; fixed raw-UniMatch FB-valid foreground; extra source-yellow subset.400x225 pixels. Future modeled covariance shared by both motion hypotheses.',
              'physical_3D_ground_truth_available': False,
              'limitations': ['Sampled internal discrepancy, not all rows/times','Flow is estimated2D; ambiguous depth/material IDs remain','Raster flow uses existing corrected first20 contributor diagnostic, not complete color-adjoint bank mass','No stress/constitutive truth or identity confirmation']}
    write_json(REPORT/'velocity_trajectory_probe.json', result)
    current = all_results['T1_association_52k']
    c = current['internal_consistency']; m = current['observed_motion_mean_over_30_pairs']
    lines = ['# 当前速度分支与形变轨迹的只读比较', '',
             f"当前T1最终52k：两分支内部相对差异{c['relative_RMSE']:.3%}，速度网络幅度仅为完整形变轨迹速度的{c['network_to_trajectory_RMS_ratio']:.3%}。鸭子观测区域的轨迹/速度推进EPE分别{m['duck_trajectory_EPE_px']:.4f}/{m['duck_velocity_EPE_px']:.4f}px，形变轨迹更符合已观测二维运动，真实三维速度与材料身份仍无真值核验。", '',
             '比较对象是网络输出的速度与完整渲染中心轨迹的导数；不同特征网格的参数不能直接相减。', '',
             result['internal_protocol'], '', result['observation_protocol'], '',
             '| 完整状态 | 内部相对RMSE | 速度/轨迹RMS | 前景轨迹EPE | 前景速度EPE | 鸭子轨迹EPE | 鸭子速度EPE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in all_results.items():
        c = r['internal_consistency']; m = r['observed_motion_mean_over_30_pairs']
        lines.append(f"|{name}|{c['relative_RMSE']:.3%}|{c['network_to_trajectory_RMS_ratio']:.3%}|{m['foreground_trajectory_EPE_px']:.6f}|{m['foreground_velocity_EPE_px']:.6f}|{m['duck_trajectory_EPE_px']:.6f}|{m['duck_velocity_EPE_px']:.6f}|")
    lines += ['', '内部误差以模型自身形变轨迹速度为分母，不是相对真实速度误差。速度单位为场景单位/归一化时间，不是m/s。',
              'EPE单位px@400x225，先逐对取均值，再30对等权平均。观测像素掩码固定，未覆盖像素不通过剔除来降低误差。黄色鸭子区域由图像颜色/alpha提取。',
              '形变轨迹与观测二维运动更一致，不证明真实三维速度、深度或材料身份。O1与其他模型拓扑可能不同，内部256行是各模型自身均匀抽样。',
              '已有修正后的前20贡献层光流诊断用于两假设的同条件比较；不是先前点库评估的完整贡献积分。检查点SHA前后不变，零优化器更新。']
    (REPORT/'velocity_trajectory_probe.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
