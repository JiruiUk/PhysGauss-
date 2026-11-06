from copy import deepcopy
import ast
import json
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.i10_temporal_memory.core import bind_memory
from test_i10_temporal_memory import fixture

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory/protocol.json').read_text())

def test_current_parent_binding_can_reject_old_nomination():
    m,_,xyz,projected,depth=fixture();before=deepcopy(m)
    assert bind_memory(m,xyz,projected,depth,P)['accepted']
    shifted=projected.copy();shifted[:,:,0]+=10
    assert not bind_memory(m,xyz,shifted,depth,P)['accepted']
    assert m==before

def test_memory_checks_preserve_declared_L4_algorithm():
    old=json.loads((ROOT/'experiments/staged_reconstruction/RubberDuck_temporal_memory/protocol.json').read_text())
    for key in ['track_weight','minimum_memory_span','memory_frame_budget','refresh_every',
                'projection_rms_max','projection_max','minimum_valid_fraction','ncc_min',
                'failures_to_revoke','passes_to_restore','visibility_min_share']:
        assert P[key]==old[key]

def test_runtime_default_off_does_not_install():
    # Inspect function body without triggering CUDA imports on CPU tests.
    tree=ast.parse((ROOT/'variants/staged_reconstruction/j7_long_memory/runtime.py').read_text())
    install=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='install')
    assert isinstance(install.body[0],ast.If) and isinstance(install.body[0].body[0],ast.Return)
    objective=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='objective')
    assert isinstance(objective.body[1],ast.If) and isinstance(objective.body[1].body[0],ast.Return)

def test_J7_protocol_keeps_full_joint_parent_and_no_growth():
    assert P['static_points']==3273 and P['dynamic_points']==15507
    assert not P['static_fully_frozen'] and P['static_geometry_frozen']
    assert P['self_depth_weight']==.1 and P['updates']==2000
    assert not P['topology_changes'] and not P['candidate_static_uses_blue_threshold']
