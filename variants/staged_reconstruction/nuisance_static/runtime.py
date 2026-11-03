"""Adapters only; original PIDG camera/rasterizer/loss code stays unchanged."""
import math
from types import SimpleNamespace
import torch
from ..train_static import PIPE,render_static
from .core import compose,support_penalty


def calibrated_camera(view,k,width,height):
    from utils.graphics_utils import getProjectionMatrix
    assert abs(k[0,2]-(width-1)/2) < 1e-6 and abs(k[1,2]-(height-1)/2) < 1e-6
    fovx = 2*math.atan(width/(2*k[0,0])); fovy = 2*math.atan(height/(2*k[1,1]))
    matrix = torch.tensor(view.T,dtype=torch.float32,device='cuda')
    projection = getProjectionMatrix(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).T.cuda()
    return SimpleNamespace(FoVx=fovx,FoVy=fovy,image_width=width,image_height=height,
        world_view_transform=matrix,full_proj_transform=matrix@projection,camera_center=matrix.inverse()[3,:3])


def layers(renderer,camera,static,nuisance=None,enabled=False):
    shared = render_static(renderer,camera,static)
    if not enabled: return {'static':shared,'nuisance':None,'composite':shared['render']}
    assert nuisance is not None
    explanation = renderer(camera,nuisance,PIPE,torch.zeros(3,device='cuda'),0.,0.,0.,override_color=nuisance.get_rgb)
    return {'static':shared,'nuisance':explanation,
            'composite':compose(shared['render'],explanation['render'],explanation['alpha'],enabled=True)}


def objective(package,target,reliable,static,protocol):
    from utils.loss_utils import ssim
    predicted = package['composite']
    photo = .8*(predicted-target).abs().mean()+.2*(1-ssim(predicted,target))
    penalty = support_penalty(package['static']['alpha'],reliable,
        package['nuisance']['alpha'] if package['nuisance'] is not None else None,
        protocol['paper_alpha_weight_static'],protocol['paper_alpha_weight_distractor'])
    return photo+penalty+protocol['static_position_reference_weight']*(static._xyz-static.initial_xyz).square().mean()
