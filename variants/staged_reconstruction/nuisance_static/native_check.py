"""Small native RGB/alpha/geometry gradient and exact disabled-path check."""
import argparse
import numpy as np
import torch
from ..prepare_data import ROOT,read,write
from ..clean_static.prepare import verify as verify_parent
from ..adaptive_static.model import AdaptiveGaussian
from ..synthetic_audit.input import load
from ..train_static import setup,render_static,optimizer_for
from .prepare import REPORT,RUNS
from .core import NuisancePlane,optimizer_for_nuisance
from .runtime import calibrated_camera,layers,objective


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-nuisance',action='store_true'); args = parser.parse_args()
    if not args.enable_nuisance: raise SystemExit('Native nuisance check is opt-in')
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    verify_parent(); torch.set_num_threads(4); protocol = read(REPORT.parent/'protocol.json')
    values = load(ROOT/protocol['synthetic_data']/'diffuse/train'); renderer = setup()
    image = values['rgb'][0]; height,width = image.shape[:2]
    camera = calibrated_camera(values['views'][0],values['K'],width,height)
    static = AdaptiveGaussian(RUNS/'diffuse/initialization.npz'); opt = optimizer_for(static)
    plane = NuisancePlane(values['views'][0],values['K'],image,count=1000); nuisance_opt = optimizer_for_nuisance(plane)
    with torch.no_grad():
        direct = render_static(renderer,camera,static); off = layers(renderer,camera,static,plane,False)
        assert torch.equal(direct['render'],off['composite'])
        assert torch.equal(direct['alpha'],off['static']['alpha']) and torch.equal(direct['depth'],off['static']['depth'])
    target = torch.from_numpy(image).permute(2,0,1).cuda()
    with np.load(RUNS/'diffuse/support.npz') as data: reliable = torch.from_numpy(data['reliable'][0]).cuda()
    initial = {n:p.detach().clone() for n,p in plane.named_parameters()}
    norms = None
    for step in range(4):
        opt.zero_grad(set_to_none=True); nuisance_opt.zero_grad(set_to_none=True)
        package = layers(renderer,camera,static,plane,True)
        package['static']['alpha'].retain_grad(); package['nuisance']['alpha'].retain_grad()
        loss = objective(package,target,reliable,static,protocol); loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in static.parameters())
        assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in plane.parameters())
        assert package['static']['alpha'].grad.abs().sum() > 0 and package['nuisance']['alpha'].grad.abs().sum() > 0
        norms = {'static':{n:float(p.grad.norm()) for n,p in static.named_parameters()},
                 'nuisance':{n:float(p.grad.norm()) for n,p in plane.named_parameters()}}
        opt.step(); nuisance_opt.step(); static.constrain(); plane.constrain()
    assert all(not torch.equal(initial[n],p) for n,p in plane.named_parameters())
    write(REPORT/'native_check.json',{'passed':True,'unchanged_native_renderer_disabled_RGB_alpha_depth_exact':True,
        'static_and_nuisance_alpha_gradients_nonzero':True,'four_native_Adam_updates':True,'gradient_norms':norms,
        'nuisance_geometry_is_camera_plane_not_physical_identity':True})
    print('NUISANCE_NATIVE_CHECK_PASSED',norms,flush=True)


if __name__ == '__main__': main()
