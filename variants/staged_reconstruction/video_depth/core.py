"""Train-only pilot windows and fixed geometry qualification rules."""
import numpy as np


ANCHORS=[2,43,86,146,188,200]
GATES={'held_static_median_max':.05,'held_static_p90_max':.15,
       'dynamic_extrapolation_fraction_max':.1,'static_support_min':1000}


def windows(train_ids,count=12):
    ids=sorted(set(map(int,train_ids)))
    if len(ids)!=len(train_ids) or not set(ANCHORS).issubset(ids):
        raise ValueError('Unique training frames must include all anchors')
    return {anchor:sorted(sorted(ids,key=lambda value:(abs(value-anchor),value))[:count]) for anchor in ANCHORS}


def check(row):
    return bool(row['median']<=GATES['held_static_median_max'] and
        row['p90']<=GATES['held_static_p90_max'] and
        row['dynamic_extrapolation_fraction']<=GATES['dynamic_extrapolation_fraction_max'] and
        row['static_support']>=GATES['static_support_min'])
