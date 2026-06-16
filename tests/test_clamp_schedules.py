import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.clamp_schedules import (
    middle_block_schedule,
    progressive_outward_schedule,
    global_schedule,
)


def test_middle_block_range():
    layers = middle_block_schedule(60)
    assert min(layers) >= 15   # 0.25 * 60
    assert max(layers) < 45    # 0.75 * 60
    assert len(layers) > 0


def test_middle_block_subset():
    for L in [10, 27, 60]:
        layers = middle_block_schedule(L)
        assert all(0 <= i < L for i in layers)


def test_global_all():
    layers = global_schedule(27)
    assert layers == list(range(27))


def test_progressive_length():
    D = np.array([0.5, 0.1, 0.8, 0.3])
    schedules = progressive_outward_schedule(D)
    assert len(schedules) == len(D)
    # First entry has 1 layer, last has all layers
    assert len(schedules[0]) == 1
    assert len(schedules[-1]) == len(D)


def test_progressive_sorted_by_gram():
    D = np.array([0.5, 0.1, 0.8, 0.3])
    schedules = progressive_outward_schedule(D)
    # Most aligned (lowest D) should come first: index 1 (D=0.1)
    assert schedules[0] == [1]


def test_progressive_cumulative():
    D = np.array([0.4, 0.2, 0.9, 0.1])
    schedules = progressive_outward_schedule(D)
    # Each set should be a superset of the previous
    for i in range(1, len(schedules)):
        prev = set(schedules[i - 1])
        curr = set(schedules[i])
        assert prev.issubset(curr), f"Step {i}: {prev} not subset of {curr}"


def test_progressive_with_indices():
    D = np.array([0.5, 0.1, 0.8])
    indices = np.array([10, 20, 30])
    schedules = progressive_outward_schedule(D, indices)
    # Most aligned is D=0.1 at original index 1, mapped to layer 20
    assert schedules[0] == [20]
    assert set(schedules[-1]) == {10, 20, 30}
