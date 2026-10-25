"""Independent read-only kinematic dashboard; never alters the frozen trainer."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time

from monitor_common import alive, tail
from monitor_isotropic import History, load
from pidg_benchmark import PROJECT, write_json, write_text_atomic

RUNS = PROJECT/'runs/pidg_kinematic_tuning'
REPORTS = PROJECT/'experiments/kinematic_tuning'
NAMES = ('V1_shared_time', 'V2_shared_time_kinematic', 'V3_frame_time', 'V4_frame_time_kinematic')
TRAIN_PHASES = ('long', 'screen', 'smoke', 'resume_check', 'flags_off')
TRAIN = re.compile(r'Training progress:[^\r\n]*?(\d+)/(\d+)\s*\[')
EVAL = re.compile(r'EVAL_PROGRESS iteration=(\d+) views=(\d+)/(\d+)')
RENDER = re.compile(r'Rendering progress:[^\r\n]*?(\d+)/(\d+)\s*\[')
METRIC = re.compile(r'Metric evaluation progress:[^\r\n]*?(\d+)/(\d+)\s*\[')


def stage_progress(phase, text):
    """Image counts belong to a separate stage, never the training iteration."""
    if phase in TRAIN_PHASES:
        train, evaluation = list(TRAIN.finditer(text)), list(EVAL.finditer(text))
        if evaluation and (not train or evaluation[-1].start() > train[-1].start()):
            item = evaluation[-1]
            if 'EVAL_METRICS ' not in text[item.end():]:
                return {'label': '训练中评估', 'iteration': int(item[1]),
                        'images': int(item[2]), 'total_images': int(item[3])}
        return {'label': '训练'}
    if phase.startswith('render_'):
        found = list(RENDER.finditer(text))
        result = {'label': '导出渲染', 'snapshot': phase.removeprefix('render_')}
    elif phase == 'metrics':
        found = list(METRIC.finditer(text))
        result = {'label': '离线图像指标评估'}
        methods = re.findall(r'Method:\s*(\S+)', text)
        if methods: result['snapshot'] = methods[-1]
    else:
        return {'label': phase or '等待启动'}
    if found:
        result.update(images=int(found[-1][1]), total_images=int(found[-1][2]))
    return result


class Monitor:
    def __init__(self, runs=RUNS, reports=REPORTS):
        self.runs, self.reports = Path(runs), Path(reports)
        self.readers, self.snapshot, self.lock, self.sequence = {}, {}, threading.Lock(), 0

    def reader(self, path, kind):
        key = (str(path), kind)
        if key not in self.readers: self.readers[key] = History(path, kind)
        return self.readers[key]

    def arm(self, name, state, active):
        directory = self.runs/name/'rubber_duck'
        train = self.reader(directory/'train.log', 'train')
        losses = train.update()
        metrics = self.reader(directory/'metric_history.jsonl', 'metric').update()
        checkpoint, info = load(directory/'checkpoints/resume.json'), load(directory/'reproduction.json')
        candidates = [(train.latest, 'training_log'), (checkpoint.get('iteration'), 'checkpoint'),
                      (metrics[-1]['iteration'] if metrics else None, 'evaluation_iteration')]
        step, source = max([(n, source) for n, source in candidates if isinstance(n, int)], default=(0, 'not_started'))
        target = info.get('target_additional_iterations', train.target or 50000)
        if name in state.get('completed', []): status = '全部完成'
        elif active.get('id') == name: status = '当前运行'
        elif step >= target: status = '训练完成'
        elif directory.exists(): status = '已初始化 / 等待'
        else: status = '排队中'
        return {'id': name, 'iteration': step, 'target': target, 'iteration_source': source,
                'checkpoint_iteration': checkpoint.get('iteration'),
                'evaluation_iteration': metrics[-1]['iteration'] if metrics else None,
                'latest_loss': losses[-1]['loss'] if losses else None,
                'fraction': min(1., step/max(1, target)), 'status': status,
                'metrics': metrics, 'run_absolute': str(directory)}

    def update(self):
        state, active = load(self.runs/'controller.json'), load(self.runs/'active.json')
        expected = self.runs/active.get('id', NAMES[0])/'rubber_duck'
        directory = Path(active.get('run', expected)).resolve()
        if not directory.is_relative_to(self.runs.resolve()):
            raise ValueError('Active path is outside this experiment')
        arms = [self.arm(name, state, active) for name in NAMES]
        current = next((row for row in arms if row['id'] == active.get('id')), arms[0])
        phase = active.get('phase', '')
        path = directory/('train.log' if phase in TRAIN_PHASES else phase+'.log')
        stage = stage_progress(phase, tail(path))
        worker_alive, controller_alive = alive(active.get('pid')), alive(state.get('pid'))
        status = state.get('status', 'waiting')
        if status == 'completed': stage = {'label': '全部实验完成'}
        elif status in ('saved_stopped', 'budget_stopped'): stage = {'label': '已保存并停止'}
        elif status == 'failed': stage = {'label': '运行失败'}
        elif controller_alive and phase == 'metrics' and not worker_alive:
            gallery = self.runs/'render_comparisons'/active.get('id', '')
            count = sum(1 for _ in (gallery/'pairs').glob('*/*.png'))
            stage = {'label': '生成 GT / Render 拼接图', 'images': count, 'total_images': 250}
        elif status in ('training', 'validating') and not controller_alive:
            status = 'worker_only' if worker_alive else 'process_exited'
            stage = {'label': '队列未运行；显示最后已记录的 iter'}
        self.sequence += 1
        payload = {'monitor_kind': 'pidg_kinematic_live', 'sequence': self.sequence,
                   'updated_utc': datetime.now(timezone.utc).isoformat(), 'refresh_seconds': 5,
                   'status': status, 'controller_alive': controller_alive, 'worker_alive': worker_alive,
                   'controller': state, 'active': active, 'stage': stage, 'current': current, 'arms': arms,
                   'total_iteration': sum(row['iteration'] for row in arms),
                   'total_target': sum(row['target'] for row in arms),
                   'log_age_seconds': round(max(0., time.time()-path.stat().st_mtime), 1) if path.exists() else None,
                   'note': 'Iterations count this refinement only. Image progress is reported separately.'}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        write_text_atomic(self.reports/'monitor_data.json', body+'\n')
        write_text_atomic(self.reports/'monitor_data.js', 'window.PIDG_KINEMATIC_DATA='+body.replace('<', '\\u003c')+';\n')
        with self.lock: self.snapshot = payload

    def loop(self):
        while True:
            start = time.monotonic()
            try: self.update()
            except Exception as error: print('Monitor read error:', error, flush=True)
            time.sleep(max(.1, 5-(time.monotonic()-start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    template = PROJECT/'scripts/kinematic_monitor.html'
    write_text_atomic(REPORTS/'monitor_live.html', template.read_text(encoding='utf-8'))
    monitor = Monitor()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path in ('/', '/monitor.html', '/monitor_live.html'):
                body, kind = template.read_bytes(), 'text/html; charset=utf-8'
            elif path in ('/api/status', '/monitor_data.json', '/monitor_data.js'):
                with monitor.lock: value = json.dumps(monitor.snapshot, ensure_ascii=False, allow_nan=False)
                if path.endswith('.js'):
                    body, kind = ('window.PIDG_KINEMATIC_DATA='+value.replace('<', '\\u003c')+';').encode('utf-8'), 'text/javascript; charset=utf-8'
                else: body, kind = value.encode('utf-8'), 'application/json; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    monitor.update()
    write_json(REPORTS/'monitor_server.json', {'pid': os.getpid(), 'url': f'http://127.0.0.1:{args.port}/',
               'refresh_seconds': 5, 'read_only_training_access': True})
    threading.Thread(target=monitor.loop, daemon=True).start()
    print(f'Kinematic monitor: http://127.0.0.1:{args.port}/', flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == '__main__': main()
