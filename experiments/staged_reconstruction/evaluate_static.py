"""Finish held-out evaluation using the original all200 RGB directory; no training."""
import numpy as np
from PIL import Image
import torch
from variants.staged_reconstruction.prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from variants.staged_reconstruction.model import observation_masks
from variants.staged_reconstruction.static_gaussian import StaticGaussian
from variants.staged_reconstruction.freeze import verify
from variants.staged_reconstruction import train_static as trainer


def evaluation_image(frame,width=400):
    path=DATA/(frame['file_path']+'.png')
    if not path.exists():path=ROOT/'data/PIDG/rubber_duck'/(frame['file_path']+'.png')
    with Image.open(path) as im:
        rgba=np.asarray(im.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR),dtype=np.float32)/255
    pos,neg=observation_masks(rgba,max(2,width//100))
    return rgba[...,:3]*rgba[...,3:4],pos,neg


def main():
    torch.set_num_threads(4);verify()
    protocol=read(REPORT/'S1_protocol.json');path=RUNS/'S1_static/checkpoint.pt';before=sha(path)
    state=torch.load(path,map_location='cpu',weights_only=False);assert state['iteration']==12000
    render=trainer.setup();g=StaticGaussian(RUNS/protocol['initialization_file']);g.load_state_dict(state['model'])
    trainer.image_data=evaluation_image
    output=ROOT/'output/staged_reconstruction/S1_static'
    val=trainer.evaluate(render,g,12000,output,'val');trainer.evaluate(render,g,12000,output,'test')
    assert sha(path)==before;verify()
    write(REPORT/'S1_evaluation_repair.json',{'training_updates_repeated':0,'checkpoint_sha256_unchanged':before,
        'frozen_training_source_unchanged':True,'evaluator_sha256':sha(__file__),
        'repair':'Original all200 image directory supplies held-out test RGB absent from training-only directory'})
    write(RUNS/'state.json',{'stage':'S1','status':'evaluated_pending_review','iteration':12000,'validation':val['metrics']})


if __name__=='__main__':main()
