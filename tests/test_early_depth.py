import pytest
from variants.staged_reconstruction.early_depth.train import schedule


@pytest.mark.parametrize('step,result',[(1,('warmup',1,800)),(12000,('warmup',12000,800)),
    (12001,('warmup',12001,1600)),(16000,('warmup',16000,1600)),
    (16001,('physics',1,1600)),(22000,('physics',6000,1600))])
def test_original_schedule_is_preserved(step,result):
    assert schedule(step)==result


def test_no_zeroth_optimizer_update():
    with pytest.raises(ValueError): schedule(0)
