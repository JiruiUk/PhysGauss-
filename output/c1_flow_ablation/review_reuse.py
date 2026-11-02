"""Static scientific figures from locked results and decoded video frames."""
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/c1_flow_ablation'
REPORTS = ROOT/'experiments/c1_flow_ablation'
NAMES = ['C1_repaired','O1_C1_flow_on']
DIRECTORIES = [ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck', ROOT/'runs/pidg_c1_flow_ablation/O1_C1_flow_on/rubber_duck']
read = lambda p: json.loads(p.read_text(encoding='utf8'))


def main():
    motion = [r for r in read(REPORTS/'motion_validation_reuse.json')['results'] if r['snapshot']=='best']
    frames = [int(Path(f['file_path']).stem) for f in read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames']]
    fig, axes = plt.subplots(1,3,figsize=(15,4.1))
    for n,d,r,color in zip(NAMES,DIRECTORIES,motion,['#3177b5','#d66a35']):
        psnr = read(d/'per_view.json')['ours_best']['PSNR']
        axes[0].plot(frames,[psnr[f'{i:05d}.png'] for i in range(50)],label=n,color=color)
        for ax,key in zip(axes[1:],['geometric_EPE_px','velocity_EPE_px']):
            ax.plot([p['source'] for p in r['per_pair']],[p[key] for p in r['per_pair']],label=n,color=color)
    for ax,title,y in zip(axes,['Test RGB reconstruction','Validation geometric motion','Validation velocity-advected motion'],['PSNR (dB), higher better','EPE (400 x 225 pixels), lower better','EPE (400 x 225 pixels), lower better']):
        ax.set(title=title,xlabel='Original source frame',ylabel=y)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Historical C1 reused, only O1 trained; validation-PSNR-selected snapshots')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)
    stats={}
    for key in ['geometric_EPE_px','velocity_EPE_px']:
        a=np.array([p[key] for p in motion[0]['per_pair']]);b=np.array([p[key] for p in motion[1]['per_pair']])
        stats[key]={'improved_pairs':int((b<a).sum()),'pairs':len(a),'mean_change':float((b-a).mean()),'median_change':float(np.median(b-a))}
    stats['finite_difference_sensitivity']={}
    for r in motion:
        stats['finite_difference_sensitivity'][r['model']]={k: max(p[k] for p in r['finite_difference_sensitivity']) for k in ['velocity_step_ratio_0.5','velocity_step_ratio_2.0','acceleration_step_ratio_0.5','acceleration_step_ratio_2.0']}
    (REPORTS/'review_statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf8')
    for n in NAMES:
        movie=OUT/n/'GT_Render_best.mp4'
        raw=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(movie),'-vf','scale=640:190','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout
        array=np.frombuffer(raw,dtype=np.uint8).reshape(50,190,640,3)
        sheet=Image.new('RGB',(1600,2000),'#14263a');draw=ImageDraw.Draw(sheet)
        for i in range(50):
            x,y=(i%5)*320,(i//5)*200
            sheet.paste(Image.fromarray(array[i,:,320:,:]),(x,y))
            draw.text((x+4,y+190),f'{n} frame {frames[i]}',fill='white')
        sheet.save(OUT/(n+'_video_all_frames.png'))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
