import pytest

import flock
from flock import FlockCollectiveMismatch, FlockUsageError, new_group


def test_scatter_world():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.rank()
        values = ["a", "b", "c", "d"] if rank == 0 else None
        return await flock.scatter(values, root=0)

    assert run() == ["a", "b", "c", "d"]


def test_scatter_subgroup():
    subset = new_group([0, 2])
    root = 0

    @flock.distribute(workers=4)
    async def run():
        rank = flock.rank()
        if rank not in subset:
            return None
        values = ["x", "y"] if rank == root else None
        return await flock.scatter(values, root=root, group=subset)

    assert run() == ["x", None, "y", None]


def test_scatter_root_mismatch():
    @flock.distribute(workers=2)
    async def run():
        rank = flock.rank()
        root = 0 if rank == 0 else 1
        values = [rank, rank] if rank == root else None
        await flock.scatter(values, root=root)
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="root"):
        run()


def test_scatter_non_root_values_raises():
    @flock.distribute(workers=2)
    async def run():
        values = [0, 1] if flock.rank() == 0 else ["unexpected"]
        await flock.scatter(values, root=0)
        return "done"

    with pytest.raises(FlockUsageError, match="only the scatter root"):
        run()


def test_scatter_root_missing_values_raises():
    @flock.distribute(workers=2)
    async def run():
        await flock.scatter(None, root=0)
        return "done"

    with pytest.raises(FlockUsageError, match="requires values on the root"):
        run()


def test_scatter_wrong_length_raises():
    @flock.distribute(workers=2)
    async def run():
        values = [0] if flock.rank() == 0 else None
        await flock.scatter(values, root=0)
        return "done"

    with pytest.raises(FlockUsageError, match="provided 1 values"):
        run()


def test_scatter_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.scatter([0], root=0)
