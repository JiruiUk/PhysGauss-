"""Loopback-only, uncached access to experiment dashboards and reports."""
from datetime import datetime,timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()

    def do_GET(self):
        if self.path=='/':
            self.send_response(302);self.send_header('Location','/integrity_tuning/monitor.html');self.end_headers();return
        super().do_GET()


if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(ROOT/'experiments')))
    host,port=server.server_address
    receipt={'pid':os.getpid(),'url':f'http://{host}:{port}/integrity_tuning/monitor.html',
        'object_id_url':f'http://{host}:{port}/object_id/monitor.html','bind':host,
        'root':str(ROOT/'experiments'),'started_utc':datetime.now(timezone.utc).isoformat(),'refresh_seconds':5}
    path=ROOT/'experiments/integrity_tuning/monitor_server.json'
    path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(receipt['url'],flush=True);server.serve_forever()
