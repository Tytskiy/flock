import pytest

import flock
from flock import FlockCollectiveMismatch, FlockUsageError, new_group


def test_all_gather_world():
    @flock.distribute(workers=4)
    async def run():
        return await flock.all_gather(f"rank {flock.get_rank()}").wait()

    assert run() == [
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
        ["rank 0", "rank 1", "rank 2", "rank 3"],
    ]


def test_all_gather_subgroup():
    group = new_group([0, 2])

    @flock.distribute(workers=4)
    async def run():
        if flock.get_rank() in group:
            return await flock.all_gather(flock.get_rank(), group=group).wait()
        return None

    assert run() == [[0, 2], None, [0, 2], None]


def test_all_gather_async():
    @flock.distribute(workers=3)
    async def run():
        future = flock.all_gather(flock.get_rank())
        await flock.isend((flock.get_rank() + 1) % flock.get_world_size(), flock.get_rank()).wait()
        return await future.wait()

    assert run() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


def test_all_gather_collective_mismatch():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.barrier().wait()
        await flock.all_gather(flock.get_rank()).wait()
        return "done"

    with pytest.raises(FlockCollectiveMismatch, match="all_gather"):
        run()


def test_all_gather_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.all_gather(0)
