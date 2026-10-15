"""Bounded recovery of a training child that stops writing progress."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


class TrainingStalled(RuntimeError):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        super().__init__(f"Training stalled; verified checkpoint iteration {checkpoint['iteration']}")


def verified_checkpoint(directory):
    path = Path(directory) / 'resume.pt'
    info = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != info['sha256'] or not info.get('optimizer_saved') or not info.get('rng_saved'):
        raise RuntimeError('Refusing restart: checkpoint checksum or completeness failed')
    return info


def run_watched(arguments, output, log, cwd, timeout=900, poll_interval=1):
    log = Path(log)
    process = subprocess.Popen(arguments, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
    previous = (log.stat().st_size, log.stat().st_mtime_ns)
    last_activity = time.monotonic()
    while True:
        try:
            return process.wait(timeout=poll_interval)
        except subprocess.TimeoutExpired:
            current = (log.stat().st_size, log.stat().st_mtime_ns)
            if current != previous:
                previous, last_activity = current, time.monotonic()
            if time.monotonic() - last_activity <= timeout:
                continue
            checkpoint = verified_checkpoint(log.parent / 'checkpoints')
            current = (log.stat().st_size, log.stat().st_mtime_ns)
            if current != previous:
                previous, last_activity = current, time.monotonic()
                continue
            if process.poll() is not None:
                return process.returncode
            # The Popen handle identifies this exact child, not a potentially reused PID.
            process.kill()
            process.wait(timeout=10)
            event = {'event':'stalled_child_stopped','pid':process.pid,'unix':time.time(),
                     'checkpoint_iteration':checkpoint['iteration'],'timeout_seconds':timeout}
            with (log.parent / 'watchdog-events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event) + '\n')
            stop_request = log.parent / 'checkpoints/save_and_stop.request'
            if stop_request.exists():
                stop_request.unlink()
                return 75
            raise TrainingStalled(checkpoint)
