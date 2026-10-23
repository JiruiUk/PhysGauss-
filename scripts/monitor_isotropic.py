"""Read-only, five-second charts for the isolated isotropic PIDG experiment."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from monitor_common import alive, gpu_status
from pidg_benchmark import PROJECT, read_json, write_json, write_text_atomic

RUNS = PROJECT / 'runs/pidg_isotropic_tuning'
REPORTS = PROJECT / 'experiments/isotropic_tuning'
T3 = PROJECT / 'runs/pidg_metric_tuning/T3_baseline_perceptual/rubber_duck'
BASELINE = PROJECT / 'runs/pidg_official/rubber_duck'
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')
PROGRESS = re.compile(r'Training progress:.*?(\d+)/(\d+)\s*\[.*?Loss=([\deE+.\-]+)')


def load(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class History:
    """Only parse new complete records; keep the latest value after a resume."""
    def __init__(self, path, kind):
        self.path, self.kind = Path(path), kind
        self.offset, self.buffer, self.rows, self.target, self.latest = 0, '', {}, None, None

    def update(self):
        try:
            if self.path.stat().st_size < self.offset:
                self.offset, self.buffer, self.rows, self.target, self.latest = 0, '', {}, None, None
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                fresh = stream.read()
                self.offset = stream.tell()
        except OSError:
            return sorted(self.rows.values(), key=lambda r: r['iteration'])
        records = re.split(r'[\r\n]', self.buffer + fresh.decode('utf-8', errors='ignore'))
        self.buffer = records.pop()
        for line in records:
            if self.kind == 'train':
                match = PROGRESS.search(line)
                if not match:
                    continue
                step, target, value = int(match[1]), int(match[2]), float(match[3])
                if not finite(value) or step < 0 or target <= 0:
                    continue
                self.target, self.latest = target, step
                self.rows[step] = {'iteration': step, 'loss': value}
            else:
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    continue
                step = value.get('iteration') if isinstance(value, dict) else None
                if not isinstance(step, int) or step < 0:
                    continue
                if self.kind == 'metric':
                    scores = value.get('metrics', {})
                    if not all(finite(scores.get(key)) for key in METRICS):
                        continue
                    self.rows[step] = {'iteration': step, 'metrics': {k: scores[k] for k in METRICS}}
                elif self.kind == 'isotropic' and finite(value.get('loss')) and finite(value.get('weighted_loss')):
                    self.rows[step] = {'iteration': step, 'loss': value['loss'], 'weighted_loss': value['weighted_loss']}
        return sorted(self.rows.values(), key=lambda r: r['iteration'])


def extrema_sample(rows, limit=1200):
    """Keep each bin's minimum and maximum, preserving loss spikes on screen."""
    if len(rows) <= limit:
        return rows
    result = [rows[0]]
    width = math.ceil((len(rows) - 2) / max(1, (limit - 2) // 2))
    for i in range(1, len(rows) - 1, width):
        bucket = rows[i:min(i + width, len(rows) - 1)]
        low = min(bucket, key=lambda r: r['loss'])
        high = max(bucket, key=lambda r: r['loss'])
        result.extend(sorted({r['iteration']: r for r in (low, high)}.values(), key=lambda r: r['iteration']))
    result.append(rows[-1])
    return result


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot = {}
        self.readers = {}
        self.sequence = 0

    def history(self, path, kind):
        key = (str(path), kind)
        if key not in self.readers:
            self.readers[key] = History(path, kind)
        return self.readers[key]

    def update(self):
        state, active = load(RUNS / 'controller.json'), load(RUNS / 'active.json')
        directory = Path(active.get('run', RUNS / 'I2_iso_1e3/rubber_duck')).resolve()
        if not directory.is_relative_to(RUNS.resolve()):
            raise ValueError('Active training path is outside the isotropic experiment')
        metrics = self.history(directory / 'metric_history.jsonl', 'metric').update()
        t3 = self.history(T3 / 'metric_history.jsonl', 'metric').update()
        training = self.history(directory / 'train.log', 'train')
        losses = training.update()
        isotropic = self.history(directory / 'isotropic_loss.jsonl', 'isotropic').update()
        metadata, config = load(directory / 'reproduction.json'), load(directory / 'metric_config.json')
        checkpoint = load(directory / 'checkpoints/resume.json')
        latest_steps = [n for n in (training.latest, checkpoint.get('iteration'), metrics[-1]['iteration'] if metrics else None) if isinstance(n, int)]
        step = max(latest_steps, default=0)
        target = training.target or metadata.get('target_additional_iterations', 50000)
        controller_alive, worker_alive = alive(state.get('pid')), alive(active.get('pid'))
        status = state.get('status', 'waiting')
        if status in ('training', 'validating') and not controller_alive:
            status = 'worker_only' if worker_alive else 'process_exited'
        log = directory / ('train.log' if active.get('phase') in ('long', 'screen', 'smoke', 'resume_check', 'flags_off') else active.get('phase', 'train') + '.log')
        self.sequence += 1
        payload = {
            'monitor_kind': 'pidg_isotropic_charts', 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'sequence': self.sequence, 'refresh_seconds': 5,
            'status': status, 'controller_alive': controller_alive, 'worker_alive': worker_alive,
            'controller': state, 'active': active, 'config': config,
            'progress': {'iteration': step, 'target': target, 'fraction': min(1, step / max(1, target)),
                         'latest_loss': losses[-1]['loss'] if losses else None},
            'log_age_seconds': round(max(0, time.time() - log.stat().st_mtime), 1) if log.exists() else None,
            'metrics': metrics, 't3_metrics': t3, 'training_loss': extrema_sample(losses),
            'isotropic_loss': isotropic, 'best': load(directory / 'metric_best.json'),
            'baseline': load(BASELINE / 'results.json').get('ours_best', {}),
            'gpu': gpu_status(), 'run_absolute': str(directory),
            'protocol': '150 train / fixed 50 official test views, 1600x900; test-selected checkpoints',
        }
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        write_text_atomic(REPORTS / 'charts_data.json', body + '\n')
        write_text_atomic(REPORTS / 'charts_data.js', 'window.PIDG_MONITOR_DATA=' + body.replace('<', '\\u003c') + ';\n')
        with self.lock:
            self.snapshot = payload

    def loop(self):
        while True:
            start = time.monotonic()
            try:
                self.update()
            except Exception as error:
                print('Chart sampling error:', error, flush=True)
            time.sleep(max(.1, 5 - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    template = PROJECT / 'scripts/isotropic_monitor.html'
    write_text_atomic(REPORTS / 'charts.html', template.read_text(encoding='utf-8'))
    monitor = Monitor()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path in ('/', '/monitor.html', '/charts.html'):
                body, kind = template.read_bytes(), 'text/html; charset=utf-8'
            elif path == '/charts_data.js':
                with monitor.lock:
                    value = json.dumps(monitor.snapshot, ensure_ascii=False, allow_nan=False)
                body, kind = ('window.PIDG_MONITOR_DATA=' + value.replace('<', '\\u003c') + ';').encode('utf-8'), 'text/javascript; charset=utf-8'
            elif path in ('/api/status', '/charts_data.json'):
                with monitor.lock:
                    body = json.dumps(monitor.snapshot, ensure_ascii=False, allow_nan=False).encode('utf-8')
                kind = 'application/json; charset=utf-8'
            elif path == '/results.md':
                body, kind = (REPORTS / 'results.md').read_bytes(), 'text/plain; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    monitor.update()
    write_json(REPORTS / 'charts_server.json', {'pid': os.getpid(), 'port': args.port,
               'url': f'http://127.0.0.1:{args.port}/', 'refresh_seconds': 5,
               'read_only_training_access': True})
    threading.Thread(target=monitor.loop, daemon=True).start()
    print(f'Isotropic charts http://127.0.0.1:{args.port}/ (refresh every 5 seconds)', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
