"""Build paired, all-test-frame report tables and scientific figures."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

COMPARE_PAGE_COUNT=13
MODELS=['H3','S0','S1','S2','S3','S4','S5','S6']
FIELDS=['PSNR','SSIM','LPIPS','dynamic_PSNR','static_PSNR']
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'tmp/pdfs/weekly_20261103/test_compare'
BASE='PIDG_Weekly_Experiment_Summary_CN_20261103'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def build():
    out=WORK/'figures';out.mkdir(parents=True,exist_ok=True)
    data={m:read(WORK/'evaluation'/f'{m}.json') for m in MODELS}
    baseline=data['H3'];ids=[r['frame'] for r in baseline['frames']]
    assert ids==list(range(1,201,4))
    for m,item in data.items():
        assert item['protocol']==baseline['protocol']
        assert item['weight_sha256']==baseline['weight_sha256']
        assert [r['frame'] for r in item['frames']]==ids
        assert item['checkpoint_unchanged_after'] and item['training_or_optimizer_updates']==0
        for i,row in enumerate(item['frames']):
            assert row['static_pixels']==baseline['frames'][i]['static_pixels']
            assert row['dynamic_pixels']==baseline['frames'][i]['dynamic_pixels']
    summary={}
    long=[]
    for m,item in data.items():
        result={}
        for k in FIELDS:
            values=[r[k] for r in item['frames']]
            if values[0] is None:result[k]=None;continue
            deltas=np.asarray(values)-np.asarray([r[k] for r in baseline['frames']])
            improvement=-deltas if k=='LPIPS' else deltas
            result[k]={'mean':float(np.mean(values)),'mean_delta':float(deltas.mean()),
                'median_delta':float(np.median(deltas)),'wins':int((improvement>1e-8).sum()),
                'ties':int((abs(improvement)<=1e-8).sum()),'losses':int((improvement< -1e-8).sum()),
                'min_delta':float(deltas.min()),'max_delta':float(deltas.max())}
        summary[m]=result
        for i,row in enumerate(item['frames']):
            entry={'model':m,'frame_id':row['frame'],'normalized_time':row['time'],
                'scope':'static only' if m=='S1' else 'full scene','checkpoint_sha256':item['checkpoint_sha256']}
            for k in FIELDS:
                entry[k]=row[k];entry['H3_'+k]=baseline['frames'][i][k]
                entry['delta_'+k]=row[k]-baseline['frames'][i][k] if row[k] is not None else None
            entry['static_pixels']=row['static_pixels'];entry['dynamic_pixels']=row['dynamic_pixels']
            long.append(entry)
    csvpath=ROOT/'output/pdf'/f'{BASE}.test_comparison.csv'
    with csvpath.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(long[0]));writer.writeheader();writer.writerows(long)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'H3':'#132e43','S0':'#a56a00','S2':'#b5aa99','S3':'#725bc2','S4':'#c8536a','S5':'#2a8b9e','S6':'#087b4a'}
    fig,axes=plt.subplots(4,1,figsize=(12,13.7),sharex=True,layout='constrained')
    configs=[('PSNR',False,'Full-frame PSNR (dB), higher is better'),
        ('dynamic_PSNR',False,'GT dynamic-region PSNR (dB), higher is better'),
        ('SSIM',True,'Full-frame SSIM difference to H3, positive is better'),
        ('LPIPS',True,'Standard VGG LPIPS difference to H3, negative is better')]
    for ax,(key,delta,title) in zip(axes,configs):
        ref=np.array([r[key] for r in baseline['frames']])
        for m in MODELS:
            if m=='S1' or (delta and m=='H3'):continue
            values=np.array([r[key] for r in data[m]['frames']])
            if delta:values-=ref
            ax.plot(ids,values,label=m,lw=2.2 if m in ['H3','S6'] else 1.15,color=colors[m],alpha=1 if m in ['H3','S6'] else .8)
        if delta:ax.axhline(0,color=colors['H3'],lw=1.1,ls='--')
        ax.set_title(title,loc='left',fontsize=12);ax.grid(alpha=.22);ax.legend(ncol=7,fontsize=9,loc='lower left')
    axes[-1].set_xlabel('Original test-frame ID (50 actual observed camera/time pairs)')
    axes[-1].set_xticks([1,21,41,61,81,101,121,141,161,181,197])
    fig.savefig(out/'all_frames.png',dpi=180);plt.close(fig)
    deltas=np.array([r['PSNR']-baseline['frames'][i]['PSNR'] for i,r in enumerate(data['S6']['frames'])])
    # Deterministic explanatory examples; neither checkpoints nor metrics are selected.
    indices=[int(deltas.argmin()),int(deltas.argmax())]
    hard=ids.index(85)
    third=hard if hard not in indices else next(int(i) for i in np.argsort(abs(deltas-np.median(deltas))) if int(i) not in indices)
    indices.append(third)
    assert len(set(indices))==3
    font=ImageFont.truetype('C:/Windows/Fonts/Dengb.ttf',31)
    selected=[]
    for index in indices:
        n=ids[index]
        with Image.open(ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png') as im:
            rgba=np.asarray(im.convert('RGBA').resize((1600,900),Image.Resampling.BILINEAR),dtype=np.float32)/255
        gt=Image.fromarray((rgba[...,:3]*rgba[...,3:4]*255).round().astype(np.uint8))
        r,g,b=gt.split();bounds=ImageChops.lighter(ImageChops.lighter(r,g),b).point(lambda x:255 if x>18 else 0).getbbox()
        x0,y0,x1,y1=bounds;crop=(max(0,x0-45),max(0,y0-45),min(1600,x1+45),min(900,y1+45))
        panels=[('GT',gt)]+[(m,Image.open(ROOT/data[m]['frames'][index]['render']).convert('RGB')) for m in ['H3','S0','S6']]
        w=500;h=round(w*(crop[3]-crop[1])/(crop[2]-crop[0]));sheet=Image.new('RGB',(w*4+12*3,h+46),'#183b50');d=ImageDraw.Draw(sheet)
        for i,(label,im) in enumerate(panels):
            x=i*(w+12);d.text((x+12,7),label,font=font,fill='white')
            sheet.paste(im.crop(crop).resize((w,h),Image.Resampling.LANCZOS),(x,46))
        dst=out/f'test_{n:04d}.png';sheet.save(dst)
        reason=['S6全幅PSNR相对H3退步最大','S6全幅PSNR相对H3差值最大',
            '历史困难原帧85' if third==hard else 'S6-H3全幅差值接近中位数'][len(selected)]
        selected.append({'frame':n,'index':index,'reason':reason,'S6_delta_PSNR':float(deltas[index]),'crop_xyxy':crop,
            'path':str(dst.relative_to(ROOT))})
    manifests={str((WORK/'evaluation'/f'{m}.json').relative_to(ROOT)):sha(WORK/'evaluation'/f'{m}.json') for m in MODELS}
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'models':data,'summary':summary,
        'baseline':'H3','S0_mapping':'S0_scratch_control historical 50k','S1_scope':'Stage S1 static-only12k',
        'protocol':baseline['protocol'],'evaluation_records_sha256':manifests,
        'selected_examples':selected,'figure':str((out/'all_frames.png').relative_to(ROOT)),
        'CSV':str(csvpath.relative_to(ROOT)),'CSV_sha256':sha(csvpath),
        'scripts_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        'no_training_updates':True,'new_checkpoint_selection':False}
    path=WORK/'comparison.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'output/pdf'/f'{BASE}.test_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'examples':selected},ensure_ascii=False,indent=2))

def prepare_comparison(root,snap):
    root,snap=Path(root),Path(snap);data=read(root/'tmp/pdfs/weekly_20261103/test_compare/comparison.json')
    assets={};record={}
    for key,rel in [('curves',data['figure'])]+[(f'example{i}',r['path']) for i,r in enumerate(data['selected_examples'])]:
        target=snap/'test_revision'/Path(rel).name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/rel).read_bytes());assets[key]=str(target.relative_to(snap))
        record[rel]={'snapshot':str(target.relative_to(snap)),'sha256':sha(target)}
    data['PDF_figure_sources']=record
    return assets,data

def render_comparison(assets,data,start,p,fig,box,table,heading,compact):
    summary=data['summary'];models=data['models'];baseline=models['H3']['frames']
    stamp=datetime.fromisoformat(data['captured_utc']).strftime('%Y-%m-%d %H:%M UTC')
    start('逐帧测试 01 / 13','相对历史H3：共同test口径',f'50张同相机/时刻test图：原帧1,5,...,197；统一复评快照：{stamp}。')
    rows=[]
    for m in MODELS:
        s=summary[m]
        def v(k,d=4):return '-' if s[k] is None else f"{s[k]['mean']:.{d}f}"
        rows.append([m+(' 静态' if m=='S1' else ''),v('PSNR',3),v('dynamic_PSNR',3),v('SSIM',5),v('LPIPS',5),
            '-' if s['PSNR'] is None else f"{s['PSNR']['wins']}/50",
            '-' if s['dynamic_PSNR'] is None else f"{s['dynamic_PSNR']['wins']}/50"])
    table(['模型','全幅dB','动态dB','SSIM','LPIPS','全幅胜H3','动态胜H3'],rows,[.8,.8,.8,.85,.85,.8,.8])
    heading('模型与公平比较边界')
    p('参考固定为历史验证晋级的H3_combined final50k；不是对每帧挑最高分。S0为历史S0_scratch_control 50k；S1是静态独立12k，S2是第三版16k，S3-S6依次继承6k/4k/4k/4k；架构、容量、相机约定和训练预算不同，因此不是单因素消融。')
    p('同一原始RGBA先双线性缩至1600×900，再黑底合成；预测RGB裁剪并取整为8bit。标准VGG LPIPS使用[-1,1]输入与相同本地权重。旧模型先合成后缩放且旧LPIPS口径不同，本节重新计算H3与S0，数值不应与前文跨表混排。','small')
    p('S1未建模鸭子，完整场景与动态指标记“-”；其每帧静态PSNR单列。动态/静态区域按既有GT外观规则统一定义，掩码对模型相同；不是独立材料ID或三维几何真值。全帧包含大量黑背景，区域分数需同时看。','small')
    s=summary['S6'];box(f"S6相对H3：全幅PSNR均值差{s['PSNR']['mean_delta']:+.3f} dB，动态区域{s['dynamic_PSNR']['mean_delta']:+.3f} dB；动态胜出{s['dynamic_PSNR']['wins']}/50帧。逐帧收益与全幅平均排名必须分开判断。")

    start('逐帧测试 02 / 13','50帧曲线：全幅与动态分开','H3固定参考；S1仅静态，未画入全幅/动态曲线。所有曲线覆盖全部50帧。')
    fig(assets['curves'],570,'图45  从上至下：全幅PSNR、动态区域PSNR、SSIM差、标准LPIPS差。差值一律为模型减H3；PSNR/SSIM差为正更好，LPIPS差为负更好。原帧横轴不是采集秒。')

    specs=[('PSNR','全画幅PSNR / dB',2),('SSIM','全画幅SSIM',4),('LPIPS','标准VGG LPIPS',5),
        ('dynamic_PSNR','共同动态区域PSNR / dB',2),('static_PSNR','共同静态区域PSNR / dB',2)]
    page=2
    for key,title,digits in specs:
        for lo in [0,25]:
            page+=1
            direction='差值为负更好' if key=='LPIPS' else '差值为正更好'
            start(f'逐帧测试 {page:02d} / 13',title+f'：{lo+1}-{lo+25}',
                '单元格上行为实测，下行为(模型-H3)；'+direction+'。S1“-”表示只训练静态。')
            rows=[]
            for i in range(lo,lo+25):
                ref=baseline[i][key];row=[str(baseline[i]['frame'])]
                for m in MODELS:
                    value=models[m]['frames'][i][key]
                    row.append('-\n仅静态' if value is None else f'{value:.{digits}f}\n({value-ref:+.{digits}f})')
                rows.append(row)
            compact(['原帧','H3参考','S0','S1静态','S2','S3','S4','S5','S6'],rows)

    start('逐帧测试 13 / 13','test图像：差值极端帧与困难帧','列：GT / H3 / S0 / S6。每一行只用GT边界确定共同裁剪，模型输出未修补。')
    for i,r in enumerate(data['selected_examples']):
        reason=r['reason']
        fig(assets[f'example{i}'],225,f"图{46+i}  原帧{r['frame']}，{reason}；S6-H3={r['S6_delta_PSNR']:+.3f} dB。展示选择规则公开，不代表平均性能；全部50帧数值见前表。")
    p('所有50帧、8模型、5类指标与H3逐帧差值均另存同名.test_comparison.csv（400行，保留全精度）。统一复评只读检查点；未训练、未基于测试集选择新模型，亦未修复图像。','small')
    return 4

if __name__=='__main__':build()
