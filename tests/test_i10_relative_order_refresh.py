"""Regression for the first actual 100-step evidence refresh failure."""
from copy import deepcopy
from experiments.staged_reconstruction.RubberDuck_relative_order.repair_run import clean_transition


def test_measurement_metadata_removed_without_mutation():
    p=dict(status='active',passes=1,failures=0,measurement={'known':True})
    before=deepcopy(p);result=clean_transition(p,True,True)
    merged=dict(**result,measurement={'known':False})
    assert p==before and result==dict(status='active',passes=2,failures=0)
    assert merged['measurement']=={'known':False}


def test_unknown_then_contradictions_and_recovery():
    s=clean_transition(None,True,True,True);assert s['status']=='active'
    for known,passed,status in [(False,False,'suspended'),(True,False,'suspended'),(True,False,'revoked'),
                                (True,True,'suspended'),(True,True,'active')]:
        s=clean_transition(dict(**s,measurement={'old':True}),known,passed)
        assert s['status']==status and 'measurement' not in s


def test_more_than_six_hundred_calls_and_periodic_merge():
    s=dict(status='active',passes=1,failures=0,measurement={'call':0})
    for call in range(1,602):
        if call%100==1:
            s=dict(**clean_transition(s,True,True),measurement={'call':call})
    assert s['status']=='active' and s['passes']==8
