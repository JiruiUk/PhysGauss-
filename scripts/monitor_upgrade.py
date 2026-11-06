"""Read-only localhost training monitor. No torch imports or trainer control actions."""
import argparse
from collections import Counter, deque
import ctypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

from pidg_benchmark import PROJECT, SCENES, completed_metrics, read_json, write_json
from snapshot_upgrade import NAMES
from upgrade_scope import selected_scenes

QUEUE = PROJECT / "variants/upgrade"
OUTPUT = QUEUE / "monitor"


def load(path, default=None):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {} if default is None else default


def tail(path, limit=1024 * 1024):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_progress(text):
    steps = list(re.finditer(r"(\d+)/(\d+)\s*\[", text))
    losses = re.findall(r"Loss=([\d.eE+-]+)", text)
    tests = re.findall(r"\[ITER (\d+)\] Evaluating test: L1 ([\d.eE+-]+) PSNR ([\d.eE+-]+)", text)
    return {
        "iteration": int(steps[-1][1]) if steps else None,
        "total": int(steps[-1][2]) if steps else None,
        "loss": float(losses[-1]) if losses else None,
        "test_psnr": [{"iteration": int(i), "value": float(p)} for i, _, p in tests],
    }


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_status():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [dict(zip(("name", "utilization", "memory_used", "memory_total", "temperature"),
                         [part.strip() for part in line.split(",")])) for line in output.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


class Monitor:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.snapshot = {}
        self.history = deque(maxlen=360)
        self.run_key = None
        self.sequence = 0

    def collect(self):
        now = time.time()
        controller, active = load(QUEUE / "controller.json"), load(QUEUE / "active.json")
        controller_live = alive(controller.get("pid"))
        child_live = alive(active.get("child_pid"))
        rows, warnings = [], []
        scenes = selected_scenes(QUEUE)
        for phase, name in NAMES.items():
            variant = PROJECT / f"variants/phase{phase:02d}_{name}"
            for scene in scenes:
                run = variant / "runs/long" / scene
                metadata_path = run / "reproduction.json"
                metadata = load(metadata_path)
                status = metadata.get("status", "unreadable" if metadata_path.exists() else "queued")
                row = {"phase": phase, "name": name, "scene": scene, "status": status,
                       "checkpoint": load(run / "checkpoints/resume.json").get("iteration")}
                if status == "completed":
                    try:
                        _, row["metrics"] = completed_metrics(run)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row.update(status="invalid_results", error=str(error))
                rows.append(row)
        current = None
        if active.get("run"):
            run = Path(active["run"]).resolve()
            # Read only files under the variant workspace, even if a sidecar is malformed.
            if run.is_relative_to((PROJECT / "variants").resolve()):
                step = active.get("step", "train")
                step = step if step in ("train", "render", "metrics") else "train"
                log = run / f"{step}.log"
                text = tail(log)
                progress = parse_progress(text if step == "train" else tail(run / "train.log"))
                checkpoint = load(run / "checkpoints/resume.json")
                age = max(0, now - log.stat().st_mtime) if log.exists() else None
                key = str(run)
                if key != self.run_key:
                    self.history.clear()
                    self.run_key = key
                if child_live and step == "train" and progress["iteration"] is not None:
                    self.history.append({"unix": now, "iteration": progress["iteration"], "loss": progress["loss"]})
                current = dict(progress, run=str(run.relative_to(PROJECT)), scene=run.name,
                               phase=run.parents[2].name, step=step, log_age_seconds=age,
                               checkpoint=checkpoint, child_alive=child_live,
                               log_tail="\n".join(text.replace("\r", "\n").splitlines()[-16:]),
                               history=list(self.history))
                if controller_live and not child_live:
                    warnings.append("当前子进程已退出，等待队列切换；若持续出现，请检查控制器日志。")
                if controller_live and child_live and age is not None and age > 900:
                    warnings.append("当前日志超过 15 分钟未更新，请检查训练状态。")
        if controller.get("status") == "running" and not controller_live:
            warnings.append("控制器记录为 running，但进程已退出。")
        self.sequence += 1
        return {"updated_at": datetime.now().astimezone().isoformat(), "unix": now,
                "sequence": self.sequence, "refresh_seconds": self.interval, "monitor_pid": os.getpid(),
                "controller": controller, "controller_alive": controller_live, "active": current,
                "counts": dict(Counter(row["status"] for row in rows)), "rows": rows,
                "scenes": scenes, "excluded_scenes": [s for s in SCENES if s not in scenes],
                "gpu": gpu_status(), "free_disk_gib": round(shutil.disk_usage(PROJECT).free / 1024**3, 1),
                "warnings": warnings}

    def update(self):
        snapshot = self.collect()
        write_json(OUTPUT / "status.json", snapshot)
        with self.lock:
            self.snapshot = snapshot

    def loop(self):
        while True:
            start = time.monotonic()
            try:
                self.update()
            except Exception as error:
                # Keep serving the previous snapshot; browser marks it stale.
                print(f"Monitor collection error: {error}", flush=True)
            time.sleep(max(.05, self.interval - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    monitor = Monitor(args.interval)
    if args.once:
        monitor.update()
        print(OUTPUT / "status.json")
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (PROJECT / "scripts/upgrade_monitor.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            elif path == "/api/status":
                with monitor.lock:
                    body = json.dumps(monitor.snapshot, ensure_ascii=False, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    monitor.update()
    threading.Thread(target=monitor.loop, daemon=True).start()
    print(f"PIDG monitor: http://127.0.0.1:{args.port}/ (every {args.interval:g}s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
