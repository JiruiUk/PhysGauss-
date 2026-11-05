"""Source composition only; no baseline renderer, model or optimizer rewrite."""
import torch


def merge_model(dynamic_model, static_source, enabled=False):
    if not enabled:
        return dynamic_model
    assert dynamic_model.keys()==static_source.keys(), 'model schema mismatch'
    for key,value in dynamic_model.items():
        other=static_source[key]
        assert value.shape==other.shape and value.dtype==other.dtype, key
    if 'centers' in dynamic_model:
        assert torch.equal(dynamic_model['centers'],static_source['centers']), 'coordinate frame mismatch'
    return {key:static_source[key] if key.startswith('static.') else value
        for key,value in dynamic_model.items()}
