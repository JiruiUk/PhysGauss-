"""CPU-only GT/H3/alpha/error panels for exposed-test diagnosis, never training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]


def main():
    report=ROOT/'experiments/temporal_repair';d=json.loads((report/'diagnosis.json').read_text(encoding='utf-8'))
    raw=ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck/test/ours_50000'
    folder=report/'diagnosis_images';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    rows=d['test_used_for_diagnosis_only'];contact=Image.new('RGB',(2400,len(rows)*490),'white')
    for j,row in enumerate(rows):
        file=f"{row['index']:05d}.png"
        images=[Image.open(raw/'gt'/file).convert('RGB'),Image.open(raw/'renders'/file).convert('RGB'),Image.open(raw/'alpha'/file).convert('RGB')]
        a,b=[np.asarray(im,dtype=np.float32)/255 for im in images[:2]];error=np.abs(a-b).mean(-1)
        heat=np.zeros((*error.shape,3),dtype=np.uint8);heat[...,0]=np.clip(error*5*255,0,255).astype(np.uint8)
        images.append(Image.fromarray(heat));tile=Image.new('RGB',(3600,750),'white');draw=ImageDraw.Draw(tile)
        for i,(name,im) in enumerate(zip(['GT','H3 render','H3 alpha','RGB error x5 (red)'],images)):
            draw.text((i*900+10,10),f"{name} / frame {row['frame']:04d}",font=font,fill='#17304a')
            tile.paste(im.crop((300,150,1200,850)),(i*900,50));im.close()
        tile.save(folder/f"test_{row['frame']:04d}.png");contact.paste(tile.resize((2400,490)),(0,j*490))
    contact.save(folder/'test_worst7_contact.jpg',quality=92)
    page='<meta charset="utf-8"><title>Worst H3 frames</title><style>body{font:16px system-ui;margin:24px}img{width:100%}</style><h1>GT | H3 | alpha | RGB error</h1><p>Exposed test frames: diagnosis only. Fixed crop x300-1200,y150-850. RGB error display clips mean absolute error x5 to[0,1].</p>'
    for row in rows:
        name=f"test_{row['frame']:04d}.png";page+=f"<h2>Frame {row['frame']} | PSNR {row['PSNR']:.3f}</h2><a href='{name}'><img src='{name}'></a>"
    (folder/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
