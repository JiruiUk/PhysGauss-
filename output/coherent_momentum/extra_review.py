"""Complete video temporal sheet and all-anchor displacement alignment panels."""
from pathlib import Path
import json
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent;ROOT=OUT.parents[1]
reader=imageio_ffmpeg.read_frames(str(OUT/'N1_coherent_momentum/GT_Render_best.mp4'),pix_fmt='rgb24')
meta=next(reader);w,h=meta['size'];assert (w,h)==(3200,950)
sheet=Image.new('RGB',(2400,1900),'white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);count=0
for i,data in enumerate(reader):
    x,y=i%5*480,i//5*190
    sheet.paste(Image.frombytes('RGB',(w,h),data).resize((480,143)),(x,y+25))
    ImageDraw.Draw(sheet).text((x+5,y+3),f'video frame {i:02d}',fill='black',font=font);count+=1
assert count==50;sheet.save(OUT/'all_video_frames.png')
data=json.loads((ROOT/'experiments/coherent_momentum/rollout_alignment_3d.json').read_text())
valid=np.array(data['records'][0]['valid'])&np.array(data['records'][1]['valid'])
fig=plt.figure(figsize=(12,6),layout='constrained');axes=[];points=[]
for i,row in enumerate(data['records']):
    ax=fig.add_subplot(1,2,i+1,projection='3d');axes.append(ax)
    current=np.array(row['x_now'])[valid]
    target=np.array(row['x_render_next'])[valid]-current
    physical=np.array(row['x_physics_next'])[valid]-current
    points.extend([target,physical])
    ax.scatter(*target.T,s=14,color='#246bc3',label='Rendered displacement')
    ax.scatter(*physical.T,s=14,color='#e46e31',marker='x',label='Stress-driven displacement')
    for a,b in zip(target,physical):ax.plot(*np.stack([a,b]).T,color='#888888',alpha=.4,linewidth=.8)
    ax.set(title=row['model'],xlabel='Delta x',ylabel='Delta y',zlabel='Delta z');ax.legend(fontsize=8)
points=np.concatenate(points);center=(points.min(0)+points.max(0))*.5;radius=np.ptp(points,axis=0).max()*.55
for ax in axes:
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius);ax.set_box_aspect((1,1,1))
fig.suptitle(f'Frame 83: all {valid.sum()} common valid anchors; local displacements in world units\nEqual axes, no amplification. This is a residual visualization, not 3D physical ground truth.',fontsize=11)
fig.savefig(OUT/'rollout_displacement_3d.png',dpi=190,bbox_inches='tight',pad_inches=.3);plt.close(fig)
print('All 50 video frames decoded; all-anchor displacement figure generated')
