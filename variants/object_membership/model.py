"""Causal evidence ledger for a single uninterrupted local image-ID domain.

Historical association is distinct from current visible support and from a
material-point identity. The caller must verify image continuity, contribution,
centers and motion before supplying a nonnegative label. Never pass component
numbers from independently seeded clips as if they were one identity domain.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    min_consecutive_observations: int = 3
    object_count: int = 4


def reciprocal_mapping(forward, reverse):
    """Return seed->component links only if both directions are unique inverses."""
    f, b = np.asarray(forward), np.asarray(reverse)
    if f.ndim != 1 or b.ndim != 1 or len(f) != len(b):
        raise ValueError('Expected equally sized one-dimensional mappings')
    if not np.issubdtype(f.dtype, np.integer) or not np.issubdtype(b.dtype, np.integer):
        raise ValueError('Mappings must use integer local IDs')
    result = np.full(len(f), -1, dtype=np.int16)
    for seed, component in enumerate(f):
        if component < 0 or component >= len(b):
            continue
        if np.count_nonzero(f == component) == 1 and b[component] == seed and np.count_nonzero(b == seed) == 1:
            result[seed] = component
    return result


def optional_membership(labels=None, frames=None, config=Config()):
    """Accumulate local evidence without filling invisible points or erasing conflicts.

    Outputs at t use only labels up to t. A contradictory supported label latches
    a conflict even when no owner had qualified yet. Missing support does not
    disprove historical object association and never counts toward qualification.
    """
    if not config.enabled:
        return None
    values = np.asarray(labels)
    numbers = np.asarray(frames)
    if config.min_consecutive_observations < 3 or config.object_count < 1:
        raise ValueError('At least three observations and one object required')
    if values.ndim != 2 or numbers.shape != (len(values),):
        raise ValueError('Expected observation-by-point labels and frame numbers')
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < -1) or np.any(values >= config.object_count):
        raise ValueError('Labels must be -1 or a valid local integer object ID')
    if not np.issubdtype(numbers.dtype, np.integer) or np.any(np.diff(numbers) <= 0):
        raise ValueError('Original observation frame numbers must increase strictly')
    n = values.shape[1]
    first = np.full(n, -1, dtype=np.int16)
    last = np.full(n, -1, dtype=np.int16)
    streak = np.zeros(n, dtype=np.int32)
    qualified = np.zeros(n, dtype=bool)
    conflict = np.zeros(n, dtype=bool)
    history = np.full(values.shape, -1, dtype=np.int16)
    visible = history.copy()
    conflicts = np.zeros(values.shape, dtype=bool)
    for t, current in enumerate(values):
        known = current >= 0
        conflict |= known & (first >= 0) & (current != first)
        new = known & (first < 0)
        first[new] = current[new]
        streak = np.where(known, np.where(current == last, streak + 1, 1), 0)
        qualified |= streak >= config.min_consecutive_observations
        history[t] = np.where(qualified & ~conflict, first, -1)
        visible[t] = np.where(known & (history[t] == current), current, -1)
        conflicts[t] = conflict
        last = current.copy()
    return {'historical_owner': history, 'visible_owner': visible, 'conflict': conflicts}
