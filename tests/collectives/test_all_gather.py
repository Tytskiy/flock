import pytest

import flock
from flock import FlockCollectiveMismatch, FlockUsageError, new_group


def test_all_gather_world():
    @flock.distribute(workers=4)
    async def run():
        return await flock.all_gather(f"rank {flock.rank()}")

    assert run() == [
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
    ]


def test_all_gather_subgroup():
    subset = new_group([0, 2])

    @flock.distribute(workers=4)
    async def run():
        if flock.rank() in subset:
            return await flock.all_gather(flock.rank(), group=subset)
        return None

    assert run() == [[0, 2], None, [0, 2], None]


def test_all_gather_async():
    @flock.distribute(workers=3)
    async def run():
        future = flock.all_gather(flock.rank())
        await flock.isend((flock.rank() + 1) % flock.world_size(), flock.rank())
        return await future

    assert run() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


def test_all_gather_collective_mismatch():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            await flock.barrier()
        await flock.all_gather(flock.rank())
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="all_gather"):
        run()


def test_all_gather_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.all_gather(0)
