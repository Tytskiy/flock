import pytest

import flock
from flock import FlockUsageError


def test_rank_and_world_size():
    world = 3

    @flock.distribute(workers=world)
    async def run():
        return flock.rank(), flock.world_size()

    assert run() == [(0, 3), (1, 3), (2, 3)]


def test_distribute_with_explicit_seed():
    world = 4

    @flock.distribute(workers=world, seed=123)
    async def run():
        rank = flock.rank()
        await flock.isend((rank + 1) % world, rank).wait()
        return await flock.recv((rank - 1) % world)

    assert run() == [(r - 1) % world for r in range(world)]


def test_distribute_requires_async():
    with pytest.raises(FlockUsageError, match="async"):

        @flock.distribute(workers=2)
        def run():
            pass


def test_distribute_rejects_async_generator():
    with pytest.raises(FlockUsageError, match="yield"):

        @flock.distribute(workers=2)
        async def run():
            yield 1


def test_distribute_without_parentheses_raises():
    with pytest.raises(FlockUsageError, match="parentheses"):

        @flock.distribute
        async def run():
            pass


@pytest.mark.parametrize("workers", [0, -1, -100])
def test_distribute_invalid_worker_count_raises(workers):
    with pytest.raises(FlockUsageError, match="greater than 0"):
        flock.distribute(workers=workers)


def test_rank_outside_distribute_raises():
    with pytest.raises(FlockUsageError):
        flock.rank()
