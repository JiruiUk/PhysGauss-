"""Display existing native PNGs side by side; no image synthesis or model edits."""
from PIL import Image, ImageDraw
from variants.staged_reconstruction.colorfree_joint.runtime import OUT, REPORT
from variants.staged_reconstruction.prepare_data import ROOT, sha, write


def main():
    files={};outputs={}
    for kind in ['static','joint']:
        canvas=Image.new('RGB',(1600,1030),'#171923');draw=ImageDraw.Draw(canvas)
        heading=('No-blue-threshold static reconstruction: source / after J6 joint training'
                 if kind=='static' else 'Fixed-view comparison: original J6 / color-free static + J6, 2000 updates')
        draw.text((12,10),heading,fill='white')
        draw.text((12,30),'Native render PNGs resized equally; no independent fixed-view geometry GT.',fill='white')
        for row,(cam,time) in enumerate([(86,200),(188,100)]):
            names=([f'initial_c{cam:03d}_colorfree_static_staticRGB_native.png',f'c{cam:03d}_t002_colorfree_static_staticRGB_native.png']
                   if kind=='static' else [f'c{cam:03d}_t{time:03d}_J6_parent_jointRGB_native.png',f'c{cam:03d}_t{time:03d}_colorfree_static_jointRGB_native.png'])
            for col,name in enumerate(names):
                path=OUT/name;files[path.relative_to(ROOT).as_posix()]=sha(path)
                label=(['Source static18k (3273 Gaussians)','Same static bank, after2000 joint updates'][col]
                       if kind=='static' else ['Original J6','Color-free fusion after2000'][col])
                y=64+row*480
                draw.text((12+800*col,y),f'c{cam} '+('static RGB' if kind=='static' else f't{time}')+' | '+label,fill='white')
                with Image.open(path) as im:canvas.paste(im.resize((800,450),Image.Resampling.LANCZOS),(800*col,y+24))
        out=OUT/(kind+'_before_after.png');assert not out.exists();canvas.save(out)
        outputs[out.relative_to(ROOT).as_posix()]=sha(out)
    write(REPORT/'presentation.json',dict(source_sha256=sha(__file__),inputs=files,outputs=outputs,
          native_RBG_resized_only=True,geometry_cleanup_not_claimed=True))


if __name__=='__main__':main()
