"""Tests for utils/seeds.py and utils/timing.py.

Owner: Member C
"""

import time
import numpy as np
import pytest

from pathsnap.utils.seeds import set_seed
from pathsnap.utils.timing import Timer, time_block


def test_set_seed_reproducibility():
    set_seed(42)
    a = np.random.rand(5)

    set_seed(42)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)


def test_timer_context_manager():
    with time_block() as t:
        time.sleep(0.05)
    assert t.seconds >= 0.04
