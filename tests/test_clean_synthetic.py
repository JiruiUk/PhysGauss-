import json
import numpy as np
import pytest
from PIL import Image
from variants.staged_reconstruction.synthetic_audit.generate import box,sphere,camera_view,trace
from variants.staged_reconstruction.synthetic_audit.input import load


def test_camera_has_positive_forward_depth_and_orthonormal_rotation():
    for a in [0,.5,2.]:
        v = camera_view(a)
        assert np.allclose(v[:3,:3]@v[:3,:3].T,np.eye(3))
        assert v[2,3] > 0 and np.linalg.det(v[:3,:3]) > .99


def test_known_surface_ray_depth_and_normal():
    origin = np.array([0.,0.,-3.]); rays = np.array([[0.,0.,1.],[2.,0.,1.]])
    d,n = box(origin,rays,[-1,-1,-1],[1,1,1])
    assert d[0] == 2 and np.isinf(d[1]) and np.array_equal(n[0],[0,0,-1])
    d,n = sphere(origin,rays,[0,0,0],1)
    assert d[0] == 2 and np.isinf(d[1]) and np.allclose(n[0],[0,0,-1])


def test_static_geometry_time_invariant_while_observed_motion_changes():
    v = camera_view(0); k = np.array([[70.,0,40],[0,70,24],[0,0,1.]])
    a = trace(v,k,80,48,0,static_only=True); b = trace(v,k,80,48,.4,static_only=True)
    assert np.array_equal(a['depth'],b['depth']) and np.array_equal(a['rgb'],b['rgb'])
    c = trace(v,k,80,48,0); d = trace(v,k,80,48,.4)
    assert np.count_nonzero(np.abs(c['rgb']-d['rgb']) > .02) > 50
    assert a['thin_static_visible'].any() and c['static_visible'].any()


def test_reflections_change_appearance_without_geometry_labels_changing():
    v = camera_view(0); k = np.array([[70.,0,40],[0,70,24],[0,0,1.]])
    a = trace(v,k,80,48,.2); b = trace(v,k,80,48,.2,glossy=True)
    assert np.array_equal(a['depth'],b['depth']) and np.array_equal(a['static_visible'],b['static_visible'])
    assert np.max(np.abs(a['rgb']-b['rgb'])) > .02


def test_input_loader_refuses_gt_fields_and_outside_rgb_path(tmp_path):
    (tmp_path/'rgb').mkdir(); Image.new('RGB',(8,4)).save(tmp_path/'rgb/a.png')
    row = {'image':'rgb/a.png','world_to_camera':np.eye(4).tolist(),'time':.2}
    meta = {'width':8,'height':4,'K':np.eye(3).tolist(),'frames':[row]}
    (tmp_path/'cameras.json').write_text(json.dumps(meta))
    assert set(load(tmp_path)) == {'rgb','views','times','K'}
    row['static_mask'] = 'forbidden.npz'; (tmp_path/'cameras.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError): load(tmp_path)
    del row['static_mask']; row['image'] = '../evaluation_only/gt/image.png'
    (tmp_path/'cameras.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError): load(tmp_path)


def test_dense_sampling_has_six_source_views_without_GT():
    from variants.staged_reconstruction.adaptive_static.geometry import select_views
    pairs = [(c,t) for c in range(32) if c%4 != 3 for t in range(5) if t != c%5]
    views = np.asarray([camera_view(2*np.pi*c/32) for c,t in pairs]); ids = [t*40 for c,t in pairs]
    assert len(pairs) == 96
    assert all(len(select_views(i,views,ids)) == 6 for i in range(len(pairs)))


def test_empty_reconstruction_cannot_win_on_completeness():
    from variants.staged_reconstruction.synthetic_audit.evaluate import score
    reference = np.array([[0.,0,0],[1.,0,0]])
    empty = score(np.zeros((0,3)),reference,reference,.05)
    assert empty['surface_precision'] is None and empty['surface_coverage'] == 0 and empty['thin_surface_coverage'] == 0
    full = score(reference,reference,reference,.05)
    assert full['surface_precision'] == full['surface_coverage'] == full['thin_surface_coverage'] == 1
