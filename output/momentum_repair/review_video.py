"""Decode the complete 50-frame video into an inspectable temporal sheet."""
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).parent
video=OUT/'M1_guarded_shared/GT_Render_best.mp4'
reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);w,h=meta['size']
assert (w,h)==(3200,950)
sheet=Image.new('RGB',(2400,1900),'white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
count=0
for i,data in enumerate(reader):
    tile=Image.frombytes('RGB',(w,h),data).resize((480,143))
    x,y=(i%5)*480,(i//5)*190;sheet.paste(tile,(x,y+25))
    ImageDraw.Draw(sheet).text((x+5,y+3),f'video frame {i:02d}',font=font,fill='black');count+=1
assert count==50
sheet.save(OUT/'all_video_frames.png')
print('Decoded and indexed all 50 video frames.')
