"""Five-second localhost monitor for the new constitutive experiment matrix."""
import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from pidg_benchmark import PROJECT,read_json,write_json
from report_constitutive import report,RUNS
from monitor_common import alive,tail,parse_progress,gpu_status


def load(path):
    try:return read_json(path)
    except (OSError,ValueError):return {}


class Monitor:
    def __init__(self):self.lock=threading.Lock();self.snapshot={};self.history=deque(maxlen=180);self.key=None;self.sequence=0
    def update(self):
        result=report();controller=load(RUNS/'controller.json');active=load(RUNS/'active.json');current=None
        if active.get('run'):
            directory=Path(active['run']).resolve()
            if directory.is_relative_to(RUNS.resolve()):
                metadata=load(directory/'reproduction.json');step=active.get('step','train')
                step=step if step in ('train','render','metrics','diagnostics') else 'train'
                log=directory/f'{step}.log';text=tail(log);progress=parse_progress(tail(directory/'train.log'))
                if self.key!=str(directory):self.history.clear();self.key=str(directory)
                if alive(active.get('child_pid')) and step=='train':self.history.append({'iteration':progress['iteration'],'loss':progress['loss']})
                current={**active,**progress,'target':metadata.get('iterations'),'profile':metadata.get('profile'),
                         'child_alive':alive(active.get('child_pid')),'checkpoint':load(directory/'checkpoints/resume.json'),
                         'log_age_seconds':round(max(0,time.time()-log.stat().st_mtime),1) if log.exists() else None,
                         'log_tail':'\n'.join(text.replace('\r','\n').splitlines()[-12:]),'history':list(self.history)}
        self.sequence+=1
        snapshot={**result,'unix':time.time(),'sequence':self.sequence,'refresh_seconds':5,'controller':controller,
                  'controller_alive':alive(controller.get('pid')),'active':current,'gpu':gpu_status(),
                  'milestones':load(PROJECT/'experiments/milestones.json')}
        write_json(RUNS/'monitor/status.json',snapshot)
        with self.lock:self.snapshot=snapshot
    def loop(self):
        while True:
            start=time.monotonic()
            try:self.update()
            except Exception as error:print('Monitor update error:',error,flush=True)
            time.sleep(max(.1,5-(time.monotonic()-start)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    monitor=Monitor()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/':body=(PROJECT/'scripts/constitutive_monitor.html').read_bytes();kind='text/html; charset=utf-8'
            elif path=='/api/status':
                with monitor.lock:body=json.dumps(monitor.snapshot,ensure_ascii=False,allow_nan=False).encode('utf-8')
                kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    monitor.update();threading.Thread(target=monitor.loop,daemon=True).start()
    print(f'Constitutive monitor http://127.0.0.1:{args.port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
