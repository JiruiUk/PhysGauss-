import importlib.util
from pathlib import Path
import numpy as np
from experiments.staged_reconstruction.RubberDuck_joint_scale.solver_v2 import fit
from variants.staged_reconstruction.i10_joint_scale.core import joint_fit

spec=importlib.util.spec_from_file_location('fixture',Path(__file__).with_name('test_i10_joint_scale.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_same_convex_objective_and_solution():
    bank,raw,norms=fixture.fixture()
    a=joint_fit(bank,norms,raw,fixture.P,enabled=True);b=fit(bank,norms,raw,fixture.P,enabled=True)
    assert a['success'] and b['success']
    assert np.allclose(a['coefficients'],b['coefficients'],atol=1e-6)
    assert abs(a['cost']-b['cost'])<1e-9


def test_held_not_read_and_default_identity():
    x=object();assert fit(None,None,None,None,original=x) is x
    bank,raw,norms=fixture.fixture();a=fit(bank,norms,raw,fixture.P,enabled=True)
    for rows in bank:
        for q in rows:
            if q['split']=='check':q['depth']*=2
    b=fit(bank,norms,raw,fixture.P,enabled=True)
    assert a==b
