import pytest
import torch
from physics.time import normalize_time, observation_time, material_time


def test_normalize_and_arbitrary_differentiable_time():
    assert normalize_time(0, 1) == 0
    assert normalize_time(9, 10) == 1
    t = torch.tensor([0.2357], dtype=torch.float64, requires_grad=True)
    result = observation_time(t)
    assert result is t
    torch.testing.assert_close(torch.autograd.grad((result**2).sum(), t)[0], 2*t)
    assert material_time(t, 112) is t
    torch.testing.assert_close(material_time(t, 112, "legacy"), t/112)
    for value in (0.2, 0.25, 0.3):
        assert observation_time(value) == value
    with pytest.raises(ValueError):
        normalize_time(0, 0)
    with pytest.raises(ValueError):
        observation_time(float("nan"))
