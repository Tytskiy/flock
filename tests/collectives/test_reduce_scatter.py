import pytest

import flock
from flock import FlockUsageError


def test_reduce_scatter():
    @flock.distribute(workers=3)
    async def run():
        values = [flock.get_rank() + 1] * 3
        return await flock.reduce_scatter(values, "sum").wait()

    assert run() == [6, 6, 6]


def test_reduce_scatter_position_mapping():
    @flock.distribute(workers=3)
    async def run():
        values = [flock.get_rank() + 10 * pos for pos in range(3)]
        return await flock.reduce_scatter(values, "sum").wait()

    assert run() == [3, 33, 63]


def test_reduce_scatter_wrong_length_raises():
    @flock.distribute(workers=3)
    async def run():
        return await flock.reduce_scatter([1, 2], "sum").wait()

    with pytest.raises(FlockUsageError, match="expected 3 values"):
        run()
