"""Read-only log, process and GPU sampling helpers."""
import ctypes
import os
from pathlib import Path
import re
import subprocess


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

