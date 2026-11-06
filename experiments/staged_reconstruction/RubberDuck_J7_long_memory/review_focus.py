"""Presentation of original native RGB/alpha/D images, no invented content."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha

B=Path(__file__).resolve().parent
O=ROOT/'output/staged_reconstruction/RubberDuck_J7_long_memory/v1'

def main():
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    names=['J7_parent','rgb_control','temporal_memory'];files={};inputs={}
    for cid,t in [(86,200),(188,100)]:
        canvas=Image.new('RGB',(1920,1240),'#171a22');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'Rubber Duck fixed c{cid} time{t}: J7 / matched continuation / memory; no independent geometry GT',font=font,fill='white')
        for row,kind in enumerate(['jointRGB','dynamicAlpha','dynamicD-alpha']):
            y=40+row*400
            draw.text((10,y),kind+'; black depth = missing' if row==2 else kind,font=small,fill='white')
            for col,name in enumerate(names):
                path=O/f'c{cid:03d}_t{t:03d}_{name}_{kind}_native.png'
                inputs[path.relative_to(ROOT).as_posix()]=sha(path)
                draw.text((col*640+10,y+20),name,font=small,fill='white')
                with Image.open(path) as im:canvas.paste(im.resize((640,360),Image.Resampling.LANCZOS),(col*640,y+40))
        path=O/f'c{cid:03d}_t{t:03d}_RGB_alpha_D_focus.png';canvas.save(path);files[path.relative_to(ROOT).as_posix()]=sha(path)
    write(B/'v1/review_focus.json',dict(files=files,original_native_inputs=inputs,source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
