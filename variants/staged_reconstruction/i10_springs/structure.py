"""Fit-selected, revocable local geometry and observed contribution support."""
from copy import deepcopy
import numpy as np
import torch
from ..i10_tracking.io import ROOT, read
from ..i10_surface_graph import runtime as graph
from ..i10_revocable_surface.core import transition, edge_measurement
from ..rubber_visibility.worker import native, colors_for
from ..train_static import camera
from ..i10_surface_support.core import covariance, support_spectrum, footprint
from .core import spring_terms, median_rest


def graph_inputs(config):
    bank = read(ROOT/config['graph_bank'])
    graph.PATCHES = bank['patches']
    graph.EDGES = bank['edges']
    graph.CONFIG = read(ROOT/config['graph_protocol'])
    return bank


def refresh_surface(renderer, scene, previous, config, initial=False):
    bank = graph_inputs(config)
    from . import runtime as a
    graph.META = a.META; graph.ASSET = a.ASSET
    centers, measurements = graph.measure(renderer, scene)
    states = []
    for i, measurement in enumerate(measurements):
        owner = bank['patches'][i].get('owner')
        if owner is not None and states[owner]['status']!='active':
            measurement = dict(measurement, passed=False, kind='unknown', confidence=0., root_not_active=True)
        states.append(transition(None if initial else previous['patches'][i], measurement, graph.CONFIG, initial))
    edges = []
    for i, edge in enumerate(bank['edges']):
        left, right = edge['i'], edge['j']
        lengths = np.linalg.norm(centers[right]-centers[left], axis=-1)
        measurement = edge_measurement(lengths, states[left], states[right], graph.CONFIG)
        edges.append(transition(None if initial else previous['edges'][i], measurement, graph.CONFIG, initial))
    if initial:
        old = read(ROOT/config['graph_health'])['controller']
        assert states == old['patches'] and edges == old['edges']
    return dict(patches=states, edges=edges)


@torch.no_grad()
def _build_parent_reference(renderer, scene, state, config):
    """Parent reference has no physical-thickness/independent-GT claim."""
    from . import runtime as a
    original = graph_inputs(config); nodes = original['patches']; edges = original['edges']
    proposals = []
    for owner, root in enumerate(nodes):
        if not root['is_anchor'] or state['patches'][owner]['status']!='active':
            continue
        members = [owner]+[j for j in root['members'] if state['patches'][j]['status']=='active']
        rows = [nodes[j]['rows'][0] for j in members]
        if len(set(rows))<config['min_surface_nodes']:
            continue
        center = np.stack([nodes[j]['parent_centers'] for j in members], 1)
        fit = graph.CONFIG['fit']
        eigen = [np.linalg.eigvalsh(np.cov(center[t], rowvar=False, bias=True))[1] for t in fit]
        if min(eigen)<config['min_tangent_variance']:
            continue
        eligible = [i for i,e in enumerate(edges) if e['owner']==owner and e['i'] in members and e['j'] in members
                    and state['edges'][i]['status']=='active']
        if not eligible:
            continue
        proposals.append(dict(owner=owner, id=root['id'], start=root['start'], nodes=members,
                              rows=rows, edge_indices=eligible, reference={}))
    with a.evidence_context(scene):
        for index, frame in enumerate(a.META['frames']):
            if a.r.stopped():
                raise a.StopRequested('User stop request during reference construction')
            active = [q for q in proposals if index-q['start'] in graph.CONFIG['fit']]
            if not active:
                continue
            dynamic = scene.dynamic_at(frame['time'])
            cov = covariance(dynamic.get_scaling, dynamic.get_rotation)
            cam = camera(frame, a.META, 400)
            for offset in range(0, len(active), 3):
                batch = active[offset:offset+3]
                packed = native(renderer, scene, cam, frame['time'], colors_for(scene, [q['rows'] for q in batch]))
                for channel, patch in enumerate(batch):
                    rows = patch['rows']; local = dynamic.get_xyz[rows]; matrices = cov[rows]
                    pairs = [(patch['nodes'].index(edges[i]['i']), patch['nodes'].index(edges[i]['j'])) for i in patch['edge_indices']]
                    uv = nodes[patch['owner']]['uv'][index-patch['start']]
                    value = footprint(packed['render'][channel], uv)
                    known = value is not None and bool(value>1e-8)
                    patch['reference'][str(index)] = dict(
                        spectrum=support_spectrum(local, matrices).cpu().tolist(),
                        max_std=torch.linalg.eigvalsh(matrices)[:, -1].clamp_min(1e-12).sqrt().cpu().tolist(),
                        edge_lengths=[float(torch.linalg.vector_norm(local[j]-local[i])) for i,j in pairs],
                        contribution_mean=float(value) if known else None,
                        coverage_reference_known=known, uv=uv)
    return dict(patches=proposals, original_all_hypotheses_retained_at=config['graph_bank'],
                parent_model_reference_not_GT=True, physical_thickness_known=False,
                summary=dict(local_patches=len(proposals), edges=sum(len(q['edge_indices']) for q in proposals),
                             rows=len({row for q in proposals for row in q['rows']}),
                             temporal_thirds=[sum(q['start']//40==b for q in proposals) for b in range(3)]))



def build_reference(renderer, scene, state, config):
    bank = _build_parent_reference(renderer, scene, state, config)
    assert bank == read(ROOT/config['surface_reference_parent']), 'Same exact J6/L3 reference must replay'
    for patch in bank['patches']:
        references = [patch['reference'][f]['edge_lengths'] for f in sorted(patch['reference'], key=int)]
        patch['spring_rest_lengths'] = median_rest(references)
    bank['fixed_stiffness'] = config['shared_stiffness']
    bank['reference_is_parent_fit_median_not_material_rest_GT'] = True
    return bank


def add_structure(renderer, scene, index, controller, config, report):
    from . import runtime as a
    bank=read(report/'surface_bank.json'); original=graph_inputs(config)
    nodes,edges=original['patches'],original['edges']; states=controller['surface']
    proposed=[p for p in bank['patches'] if str(index) in p['reference']]
    zero=scene.pools[0]._xyz.sum()*0; total=zero;count=patches=0
    if not proposed:return zero,dict(surface_patches=0,surface_edges=0,coverage_groups=0)
    dynamic=scene.dynamic_at(a.META['frames'][index]['time'])
    for patch in proposed:
        t=index-patch['start']
        if any(states['patches'][j]['status']!='active' or not states['patches'][j]['measurement']['frames'][t]['passed'] for j in patch['nodes']):continue
        if any(states['edges'][j]['status']!='active' for j in patch['edge_indices']):continue
        root=dynamic.get_xyz[nodes[patch['owner']]['rows'][0]]
        view=root.new_tensor(a.ASSET['views'][index]); k=root.new_tensor(a.ASSET['k'])
        point=root@view[:3,:3].T+view[:3,3];pixel=point@k.T
        error=pixel[:2]/pixel[2].clamp_min(1e-8)-root.new_tensor(patch['reference'][str(index)]['uv'])
        if not bool(torch.isfinite(error).all() and point[2].detach()>config['depth_min'] and torch.linalg.vector_norm(error.detach())<=config['projection_max']):continue
        pairs=[(patch['nodes'].index(edges[i]['i']),patch['nodes'].index(edges[i]['j'])) for i in patch['edge_indices']]
        term=spring_terms(dynamic.get_xyz[patch['rows']],pairs,patch['spring_rest_lengths'],config['shared_stiffness'])
        confidence=min(states['patches'][j]['confidence'] for j in patch['nodes'])
        total=total+confidence*term['energy'];patches+=1;count+=len(pairs)
    return config['structural_weight']*total/max(len(proposed),1),dict(surface_patches=patches,surface_edges=count,coverage_groups=0)
