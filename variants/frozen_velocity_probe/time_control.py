"""Post-hoc, same-label/same-budget monotonic-time diagnostic."""
import argparse
import json
import math
from pathlib import Path
import time
from .run import ROOT,REPORT,initialize,read,write,sha,datasets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=REPORT/'clock_control.json');args=parser.parse_args()
    control=read(args.config)
    if not control.get('enabled',False):print('disabled');return
    np,torch,VelocityNet=initialize()
    from .core import velocity_parameters,alignment_metrics,normalized_mse,state_hash
    from .clock import apply_clock
    base=read(REPORT/'config.json');out=ROOT/base['output_run'];manifest=read(out/'manifest.json')
    assert sha(out/'targets.pt')==manifest['targets_sha256']
    data=torch.load(out/'targets.pt',map_location='cpu',weights_only=True);sets=datasets(data,torch);fit=sets['known_train']
    torch.manual_seed(control['seed']);torch.cuda.manual_seed_all(control['seed'])
    net=VelocityNet().cuda();net.load_state_dict(data['source_velocity'])
    velocity_parameters(net);apply_clock(net,control.get('clock_mode','legacy'))
    params=[p for p in net.parameters() if p.requires_grad];optimizer=torch.optim.Adam(params,lr=control['lr'])
    scale=fit['target'].square().mean().sqrt();threshold=scale.item()*.1
    gen=torch.Generator(device='cuda').manual_seed(control['seed'])
    dest=out/control['name'];dest.mkdir(exist_ok=True)
    if (dest/'history.jsonl').exists():raise ValueError('Refusing to overwrite existing control')
    source_receipt={str(p.resolve().relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('clock.py'),args.config]}
    head_hash=state_hash(net.stress_head);held=data['particle_indices'][data['ntrain']:].unique().cuda()
    held_before=net.index_embed.weight[held].detach().clone();first_grads={};start=time.monotonic()
    @torch.no_grad()
    def evaluate(s):
        pred=torch.cat([net(s['coords'][i:i+8192],s['ids'][i:i+8192])[0] for i in range(0,len(s['coords']),8192)])
        return alignment_metrics(pred,s['target'],threshold)
    def record(step,loss=None):
        row={'step':step,'seconds':time.monotonic()-start,'loss':loss,'metrics':{k:evaluate(s) for k,s in dict(fit=fit,**sets).items()}}
        with (dest/'history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(control['name'],'step',step,'fit',round(row['metrics']['fit']['relative_rmse'],5),
              'time-val',round(row['metrics']['known_validation']['relative_rmse'],5),
              'cross',round(row['metrics']['heldout_validation']['relative_rmse'],5),'sec',round(row['seconds'],1),flush=True)
        return row
    initial=record(0)['metrics']
    for step in range(1,control['steps']+1):
        ix=torch.randint(len(fit['coords']),(control['batch_size'],),generator=gen,device='cuda')
        for group in optimizer.param_groups:group['lr']=control['lr']*(.1+.9*(1+math.cos(math.pi*(step-1)/control['steps']))/2)
        optimizer.zero_grad(set_to_none=True)
        loss=normalized_mse(net(fit['coords'][ix],fit['ids'][ix])[0],fit['target'][ix],scale)
        assert torch.isfinite(loss);loss.backward()
        if step==1 or step%1000==0:
            for name,p in net.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(),name
                    if step==1:first_grads[name]=p.grad.norm().item()
        optimizer.step()
        if step%control['evaluation_interval']==0 or step==control['steps']:final=record(step,loss.item())
    assert state_hash(net.stress_head)==head_hash and torch.equal(held_before,net.index_embed.weight[held])
    assert sha(out/'targets.pt')==manifest['targets_sha256']
    assert all(sha(ROOT/p)==v for p,v in manifest['source_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in source_receipt.items())
    torch.save({'velocity_net':{k:v.cpu() for k,v in net.state_dict().items()},'arm':control,'diagnostic_only':True,
                'target_sha256':manifest['targets_sha256'],'clock_mode':control['clock_mode']},dest/'velocity_only.pt')
    value=dict(arm=control,initial=initial,final=final,gradient_norms_first_step=first_grads,
       source_sha256=source_receipt,targets_unchanged=True,source_hashes_unchanged=True,
       heldout_id_embedding_bitwise_unchanged=True,unused_stress_head_parameters_unchanged=True)
    write(dest/'result.json',value);write(REPORT/'clock_control_result.json',value)
    print('CONTROL COMPLETE',flush=True)


if __name__=='__main__':main()
