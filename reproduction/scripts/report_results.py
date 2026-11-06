"""Collect real run state and metrics without presenting incomplete training as reproduction."""
import argparse
import ctypes
import csv
import io
import json
from pathlib import Path
import re
import time

SCENES = ["mechanics_cloth", "balls_reaction", "dry_ice", "motion_kuro", "rubber_duck"]
PAPER = {"mechanics_cloth": [39.96, 0.991, 0.034], "balls_reaction": [33.31, 0.988, 0.031],
         "dry_ice": [26.12, 0.926, 0.138], "motion_kuro": [26.97, 0.953, 0.051],
         "rubber_duck": [28.43, 0.976, 0.037]}


def controller_alive(pid):
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        result = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(result))) and result.value == 259
    finally:
        kernel.CloseHandle(handle)


def collect(root):
    rows = []
    for scene in SCENES:
        path = root / scene
        info_path = path / "reproduction.json"
        row = {"scene": scene, "status": "queued", "step": 0, "PSNR": "", "SSIM": "", "LPIPS": "",
               "paper_PSNR": PAPER[scene][0], "paper_SSIM": PAPER[scene][1], "paper_LPIPS": PAPER[scene][2]}
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                info = {}
            row["status"] = info.get("status", "pending")
            if (path / "train.log").is_file():
                with (path / "train.log").open("rb") as stream:
                    stream.seek(max(0, (path / "train.log").stat().st_size-50000))
                    text = stream.read().decode("utf-8", errors="replace")
                steps = re.findall(r"(\d+)/50000", text)
                row["step"] = int(steps[-1]) if steps else 0
                row["log_age_seconds"] = round(time.time() - (path / "train.log").stat().st_mtime)
            if row["status"] == "completed":
                result = info["metrics"]["ours_best"]
                row.update({k: result[k] for k in ("PSNR", "SSIM", "LPIPS")})
                row["step"] = info["iterations"]
            if row["status"] == "saved_stopped":
                row["step"] = info["checkpoint_iteration"]
            if row["status"] == "failed":
                row["error"] = info.get("error", "See training log")
        rows.append(row)
    return rows


def write_report(project, root, mirror=None):
    rows = collect(root)
    pause_path = project / "reproduction/pause-state.json"
    if pause_path.is_file() and json.loads(pause_path.read_text(encoding="utf-8")).get("state") == "paused":
        for row in rows:
            if row["status"] in {"training", "rendering", "evaluating"}:
                row["status"] = "paused"
    controller_path = project / "reproduction/controller.json"
    if controller_path.is_file():
        controller = json.loads(controller_path.read_text(encoding="utf-8"))
        if not controller_alive(controller["pid"]):
            for row in rows:
                if row["status"] == "queued":
                    row["status"] = "not_run"
                elif row["status"] in {"training", "rendering", "evaluating", "pending"}:
                    row["status"] = "incomplete"
                    row["error"] = "Controller exited before this scene completed; inspect the run logs."
    for row in rows:
        if row["status"] == "training" and row.get("log_age_seconds", 0) > 600:
            row["status"] = "stalled"
            row["error"] = "Training log has not advanced for over 10 minutes; process liveness alone does not confirm progress."
    summary = {"run_root": str(root), "profile": "patched_official_release", "rows": rows,
               "all_completed": all(r["status"] == "completed" for r in rows), "updated_unix": time.time()}
    output = project / "reproduction/results_summary.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(output)
    columns = ["scene", "status", "step", "PSNR", "SSIM", "LPIPS", "paper_PSNR", "paper_SSIM", "paper_LPIPS"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (project / "reproduction/results_summary.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
    text = "# PIDG 复现进度与实测结果\n\n"
    text += f"项目：`{project}`。独立 Conda 环境：`envs/pidg-cu128`。\n\n"
    text += "已接入官方动态高斯、4D hash、材料场、CMR、光流 CUDA 渲染及正式 evaluator。五个场景的必需文件已下载并校验。\n\n"
    text += "RTX 5070 Ti / CUDA 12.8 上的三项 GPU 梯度预检及真实 Mechanics Cloth 的 120 步训练→渲染→LPIPS 评估已通过。短训练只验证工程闭环，不代表论文复现完成。\n\n"
    text += "正式首轮为公开代码设置：50,000 步、1600 像素宽度、seed0。与论文 seed42、CMR 权重等差异已在项目 `reproduction/README.md` 记录。\n\n"
    text += "| 场景 | 状态 | 训练步数 | 本机 PSNR | 本机 SSIM | 本机 LPIPS | 论文 PSNR |\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        def value(key):
            return f"{row[key]:.5f}" if isinstance(row[key], (int, float)) else "待完成"
        text += f"| {row['scene']} | {row['status']} | {row['step']}/50000 | {value('PSNR')} | {value('SSIM')} | {value('LPIPS')} | {row['paper_PSNR']} |\n"
    text += "\n只有完成 train→render→metrics 的场景才写入本机正式指标；未完成处保留为空。论文参考值来自 [Table 1](https://arxiv.org/html/2511.06299v1)，不作为本机结果。\n\n"
    text += "训练日志、模型与图像保存于 `runs/pidg_official/<scene>`。实时文件 `reproduction/results_summary.json`、`results_summary.csv` 汇总状态。\n"
    if any(row["status"] == "paused" for row in rows):
        text += "\n训练和后续队列已按用户要求挂起。进程内 optimizer 状态、内存和显存保留；恢复时可原地继续。\n"
    if any(row["status"] == "saved_stopped" for row in rows):
        text += "\n已保存完整模型、两组优化器、采样顺序和随机状态，随后结束训练进程以释放显存。下次通过 -Resume 载入磁盘断点，无需保留进程。\n"
    for row in rows:
        if "error" in row:
            text += f"\n{row['scene']} 运行错误：{row['error']}\n"
    (project / "reproduction/results_summary.md").write_text(text, encoding="utf-8")
    if mirror:
        mirror = Path(mirror)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(text, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--mirror")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    root = project / "runs/pidg_official"
    while True:
        summary = write_report(project, root, args.mirror)
        if not args.follow or all(r["status"] in {"completed", "failed", "not_run", "incomplete", "saved_stopped"} for r in summary["rows"]):
            print(json.dumps(summary, indent=2), flush=True)
            break
        time.sleep(30)
