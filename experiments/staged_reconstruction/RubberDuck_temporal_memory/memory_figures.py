"""Display observed track memory coverage; never create training assets."""
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,write,sha

def main():
    b=Path(__file__).parent;bank=read(b/'v1/bank.json');meta=read(DATA/'transforms_train.json')
    out=ROOT/'output/staged_reconstruction/RubberDuck_temporal_memory/v1';out.mkdir(parents=True,exist_ok=True)
    active={m['id'] for m in bank['memories'] if m['binding']['accepted']}
    long=sorted([m for m in bank['memories'] if len(m['frames'])>=16],key=lambda m:(m['frames'][0],m['id']))
    image=Image.new('RGB',(1200,110+13*len(long)),'#171923');draw=ImageDraw.Draw(image)
    draw.text((12,10),'L4 actual RGB track memory: green=native admitted; gray=unknown/unbound',fill='white')
    draw.text((12,28),'730 tracklets ->75 cross-window memories ->22 admitted; no object identity / independent 3D GT',fill='white')
    for f in [0,40,80,119]:
        x=210+f*8;draw.line((x,55,x,image.height-20),fill='#394052');draw.text((x,48),str(f),fill='white')
    for j,m in enumerate(long):
        y=78+j*13;color='#4cdd96' if m['id'] in active else '#7d8495'
        draw.text((12,y-4),m['id']+' '+str(len(m['frames']))+' frames',fill=color)
        for f in m['frames']:draw.ellipse((210+f*8-2,y-2,210+f*8+2,y+2),fill=color)
    timeline=out/'memory_coverage.png';image.save(timeline)
    chosen=[]
    for band in range(3):
        values=[m for m in long if m['frames'][0]//40==band]
        if values:chosen.append(values[0])
    canvas=Image.new('RGB',(1600,270*len(chosen)),'#171923')
    for j,m in enumerate(chosen):
        picks=[0,len(m['frames'])//3,2*len(m['frames'])//3,len(m['frames'])-1]
        for column,i in enumerate(picks):
            f=m['frames'][i];frame=meta['frames'][f];path=DATA/(frame['file_path']+'.png')
            if not path.exists():path=ROOT/'data/PIDG/rubber_duck/train'/path.name
            arr=np.asarray(Image.open(path).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            tile=Image.fromarray(np.uint8(arr[...,:3].astype(np.float32)*arr[...,3:4]/255))
            d=ImageDraw.Draw(tile);x,y=m['uv'][i];color='#4cdd96' if m['id'] in active else '#ffb164'
            d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
            canvas.paste(tile,(column*400,j*270+38));d=ImageDraw.Draw(canvas)
            d.text((column*400+5,j*270+5),m['id']+' train index '+str(f),fill='white')
            d.text((column*400+5,j*270+21),'admitted' if m['id'] in active else 'unknown, retained',fill=color)
    montage=out/'observed_memory_examples.png';canvas.save(montage)
    write(b/'v1/memory_figures.json',dict(eval_only=True,selected_ids=[m['id'] for m in chosen],
        figures={p.relative_to(ROOT).as_posix():sha(p) for p in [timeline,montage]},source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
