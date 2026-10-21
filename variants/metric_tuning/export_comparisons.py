"""Wait for the search, then make a GT / metric-winner gallery without GPU work."""
import argparse
from datetime import datetime, timezone, timedelta
import html
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from pidg_benchmark import read_json,write_json,write_text_atomic
from monitor_common import alive
REPORTS=ROOT/'experiments/metric_tuning'
RUNS=ROOT/'runs/pidg_metric_tuning'
METRICS=('PSNR','SSIM','LPIPS','MS-SSIM')


def export():
    from PIL import Image,ImageDraw,ImageFont
    winners=read_json(REPORTS/'results.json')['best_by_metric']
    output=RUNS/'render_comparisons';output.mkdir(exist_ok=True)
    paths={};entries=[];table=[]
    for key,row in winners.items():
        raw=Path(row['run'])/('test/ours_'+row['snapshot'])
        expected={f'{i:05d}.png' for i in range(50)}
        assert all({p.name for p in (raw/kind).glob('*.png')}==expected for kind in ('renders','gt'))
        actual=read_json(Path(row['run'])/'results.json')['ours_'+row['snapshot']]
        assert all(abs(actual[m]-row['metrics'][m])<1e-4 for m in METRICS)
        paths[key]={'renders_absolute':str(raw/'renders'),'gt_absolute':str(raw/'gt')}
        entries.append((key+': '+row['id'],raw/'renders/00025.png',raw/'gt/00025.png'))
        table.append('<tr><td>'+'</td><td>'.join(html.escape(str(v)) for v in
                     [key,row['id'],row['additional_iteration'],*[f"{row['metrics'][m]:.6f}" for m in METRICS]])+'</td></tr>')
    gt=Image.open(entries[0][2]).convert('RGB');assert gt.size==(1600,900)
    for name in sorted(expected):
        reference=Image.open(Path(paths['PSNR']['gt_absolute'])/name).convert('RGB')
        assert reference.size==(1600,900)
        for raw in paths.values():
            assert Image.open(Path(raw['gt_absolute'])/name).convert('RGB').tobytes()==reference.tobytes()
    panels=[('GT',gt)]+[(label,Image.open(pred).convert('RGB')) for label,pred,_ in entries]
    image=Image.new('RGB',(1984,912),'#101725');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',24)
    draw.text((16,14),'Rubber Duck | fixed official test index25 | GT + four metric winners',font=font,fill='#e5edf8')
    for i,(label,panel) in enumerate(panels):
        x=16+i%3*656;y=68+i//3*416
        draw.text((x,y),label,font=font,fill='#83e1c0')
        image.paste(panel.resize((640,360),Image.Resampling.LANCZOS),(x,y+36))
    image.save(output/'gt_and_metric_winners.png')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>PIDG metric winners</title><style>body{max-width:1400px;margin:30px auto;padding:0 22px;background:#101725;color:#e5edf8;font:16px/1.6 system-ui}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #344255}a{color:#83e1c0}</style><h1>GT + four independently selected metric winners</h1><p>Fixed 50 official test views. Historical test-based selection. Different metrics may choose the same model; unchanged reference models remain eligible.</p><img src="gt_and_metric_winners.png"><table><tr><th>Objective</th><th>Model</th><th>Extra step</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MS-SSIM</th></tr>'
    page+=''.join(table)+'</table><p>Display image uses a common test view and resizing. Metrics use full 1600x900 PNGs.</p><p><a href="manifest.json">All raw render / GT absolute directories (50 views each)</a></p></html>'
    write_text_atomic(output/'index.html',page)
    manifest={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
              'index_absolute':str(output/'index.html'),'preview_absolute':str(output/'gt_and_metric_winners.png'),
              'test_frame_count':50,'preview_test_index':25,'winners':winners,'raw_paths':paths,'matched_gt_verified':True}
    write_json(output/'manifest.json',manifest);write_json(REPORTS/'render_comparisons.json',manifest)
    path=REPORTS/'results.md';text=path.read_text(encoding='utf-8').split('<!-- final-render-comparisons -->')[0].rstrip()
    write_text_atomic(path,text+'\n\n<!-- final-render-comparisons -->\n## GT / metric-winner renders\n\n[Open gallery](../../runs/pidg_metric_tuning/render_comparisons/index.html) · [Absolute raw paths](render_comparisons.json)\n')
    print(str(output/'index.html'),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--controller-pid',type=int);args=parser.parse_args()
    if args.wait:
        while True:
            state=read_json(RUNS/'controller.json')
            if args.controller_pid and state.get('pid')!=args.controller_pid:
                time.sleep(2);continue
            if state['status'] in ('failed','saved_stopped'):
                write_json(REPORTS/'comparison_export_status.json',{'status':'not_exported','controller_status':state['status']});return
            if state['status'] in ('completed','budget_complete') and not alive(state['pid']):break
            if datetime.now(timezone.utc)>datetime.fromisoformat(state['deadline_utc'])+timedelta(hours=1):
                raise RuntimeError('Controller exceeded its deadline window')
            time.sleep(10)
    export()


if __name__=='__main__':main()
