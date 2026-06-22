import pytest

import flock
from flock import FlockCollectiveMismatch, FlockUsageError, new_group


def test_scatter_world():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        values = ["a", "b", "c", "d"] if rank == 0 else None
        return await flock.scatter(values, src=0).wait()

    assert run() == ["a", "b", "c", "d"]


def test_scatter_subgroup():
    src = 0

    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        group = await new_group([0, 2])
        if rank not in group:
            return None
        values = ["x", "y"] if rank == src else None
        return await flock.scatter(values, src=src, group=group).wait()

    assert run() == ["x", None, "y", None]


def test_scatter_src_mismatch():
    @flock.distribute(workers=2)
    async def run():
        rank = flock.get_rank()
        src = 0 if rank == 0 else 1
        values = [rank, rank] if rank == src else None
        await flock.scatter(values, src=src).wait()
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="src"):
        run()


def test_scatter_non_src_values_raises():
    @flock.distribute(workers=2)
    async def run():
        values = [0, 1] if flock.get_rank() == 0 else ["unexpected"]
        await flock.scatter(values, src=0).wait()
        return "done"

    with pytest.raises(FlockUsageError, match="only the scatter src"):
        run()


def test_scatter_src_missing_values_raises():
    @flock.distribute(workers=2)
    async def run():
        await flock.scatter(None, src=0).wait()
        return "done"

    with pytest.raises(FlockUsageError, match="requires values on the src"):
        run()


def test_scatter_wrong_length_raises():
    @flock.distribute(workers=2)
    async def run():
        values = [0] if flock.get_rank() == 0 else None
        await flock.scatter(values, src=0).wait()
        return "done"

    with pytest.raises(FlockUsageError, match="provided 1"):
        run()


def test_scatter_src_not_in_group_raises():
    @flock.distribute(workers=3)
    async def run():
        rank = flock.get_rank()
        group = await new_group([0, 1])
        if rank not in group:
            return None
        await flock.scatter(None, src=2, group=group).wait()
        return "done"

    with pytest.raises(FlockUsageError, match="not in the group"):
        run()


def test_scatter_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.scatter([0], src=0)
