"""Suspend/resume only the verified PIDG process instances recorded for this run."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
native = ctypes.WinDLL("ntdll")
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
native.NtSuspendProcess.argtypes = native.NtResumeProcess.argtypes = [wintypes.HANDLE]
native.NtSuspendProcess.restype = native.NtResumeProcess.restype = ctypes.c_long


def control(record, resume=False):
    command = record["command"].lower()
    if "physgauss" not in command or not any(x in command for x in ("run_baseline.py", "entrypoint.py train_pidg.py", "report_results.py")):
        raise ValueError("Process is outside the recorded PIDG run")
    handle = kernel.OpenProcess(0x1800, False, record["pid"])
    if not handle:
        raise OSError(ctypes.get_last_error(), f"Unable to open PID {record['pid']}")
    try:
        creation, exit_time, cpu_kernel, cpu_user = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in (creation, exit_time, cpu_kernel, cpu_user))):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        timestamp = ((creation.dwHighDateTime << 32) | creation.dwLowDateTime)/1e7 - 11644473600
        if abs(timestamp-datetime.fromisoformat(record["created"]).timestamp()) > 0.002:
            raise ValueError(f"PID {record['pid']} was reused; refusing to control it")
        result = (native.NtResumeProcess if resume else native.NtSuspendProcess)(handle)
        if result < 0:
            raise OSError(f"Process control failed with NTSTATUS {result & 0xffffffff:08x}")
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pause", "resume"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state_path = root / "pause-state.json"
    if args.action == "pause":
        if state_path.exists() and json.loads(state_path.read_text()) .get("state") == "paused":
            raise RuntimeError("Already paused; avoiding nested suspend counts")
        state = json.loads((root / "pause-targets.json").read_text(encoding="utf-8-sig"))
        records = sorted(state["processes"], key=lambda r: 0 if "run_baseline.py" in r["command"] else 1 if "entrypoint.py" in r["command"] else 2)
        suspended = []
        try:
            for record in records:
                control(record)
                suspended.append(record)
                print(f"Suspended PID {record['pid']}", flush=True)
        except Exception:
            for record in reversed(suspended):
                control(record, resume=True)
            raise
        state.update(state="paused", processes=suspended, suspended_at_utc=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    else:
        state = json.loads(state_path.read_text())
        if state["state"] != "paused":
            raise RuntimeError("No paused run is recorded")
        for record in reversed(state["processes"]):
            control(record, resume=True)
            print(f"Resumed PID {record['pid']}", flush=True)
        state.update(state="running", resumed_at_utc=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
