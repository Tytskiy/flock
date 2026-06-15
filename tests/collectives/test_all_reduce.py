import pytest

import flock
from flock import FlockCollectiveMismatch, FlockUsageError, new_group


def test_all_reduce_sum():
    @flock.distribute(workers=4)
    async def run():
        return await flock.all_reduce(flock.get_rank(), "sum").wait()

    assert run() == [6, 6, 6, 6]


def test_all_reduce_max_subgroup():
    group = new_group([0, 2, 3])

    @flock.distribute(workers=4)
    async def run():
        if flock.get_rank() in group:
            return await flock.all_reduce(flock.get_rank(), "max", group=group).wait()
        return None

    assert run() == [3, None, 3, 3]


def test_all_reduce_collective_mismatch():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.barrier().wait()
        await flock.all_reduce(flock.get_rank(), "sum").wait()
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="all_reduce"):
        run()


def test_all_reduce_op_mismatch():
    @flock.distribute(workers=2)
    async def run():
        op = "sum" if flock.get_rank() == 0 else "max"
        await flock.all_reduce(flock.get_rank(), op).wait()
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="op"):
        run()


def test_all_reduce_incompatible_values_raises():
    @flock.distribute(workers=2)
    async def run():
        value = 1 if flock.get_rank() == 0 else "x"
        return await flock.all_reduce(value, "sum").wait()

    with pytest.raises(FlockUsageError, match="could not combine"):
        run()


def test_all_reduce_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.all_reduce(0, "sum")
