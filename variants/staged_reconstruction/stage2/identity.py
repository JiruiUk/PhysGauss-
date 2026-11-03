"""Adjacent reciprocal image evidence; component numbers never imply physical ID."""
import numpy as np
import torch
from output.surface_support.core import propagate


def directional(source, target, forward, backward, minimum=30, purity=.95, coverage=.25):
    """Map source components into target only with substantial unmixed support."""
    moved, _ = propagate(source, target, forward, backward)
    count = int(source.max()) + 1
    result = np.full(count, -1, np.int16)
    candidates = [[] for _ in range(count)]
    for k in range(int(target.max()) + 1):
        region = target == k
        values = moved[region]; values = values[values >= 0]
        if not len(values): continue
        ids, amounts = np.unique(values, return_counts=True)
        best = int(np.argmax(amounts)); seed = int(ids[best]); support = int(amounts[best])
        if support >= minimum and support >= coverage * region.sum() and support >= purity * len(values):
            candidates[seed].append(k)
    for seed, targets in enumerate(candidates):
        if len(targets) == 1: result[seed] = targets[0]
    return result


def reciprocal(forward, backward):
    """Variable component counts are allowed, but merges/splits are not ID links."""
    result = np.full(len(forward), -1, np.int16)
    for seed, target in enumerate(forward):
        if 0 <= target < len(backward) and backward[target] == seed and np.sum(forward == target) == 1 and np.sum(backward == seed) == 1:
            result[seed] = target
    return result


def advance(previous, adjacent):
    """A lost original identity remains unknown throughout this domain."""
    current = np.full_like(previous, -1)
    for seed, component in enumerate(previous):
        if 0 <= component < len(adjacent): current[seed] = adjacent[component]
    return current
