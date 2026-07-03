import pytest

import flock
from flock import FlockUsageError


def test_rank_and_get_world_size():
    world = 3

    @flock.distribute(workers=world)
    async def run():
        return flock.get_rank(), flock.get_world_size()

    assert run() == [(0, 3), (1, 3), (2, 3)]


def test_distribute_with_explicit_seed():
    world = 4

    @flock.distribute(workers=world, seed=123)
    async def run():
        rank = flock.get_rank()
        await flock.isend((rank + 1) % world, rank).wait()
        return await flock.recv((rank - 1) % world)

    assert run() == [(r - 1) % world for r in range(world)]


def test_distribute_with_fifo_policy():
    world = 3

    @flock.distribute(workers=world, policy=flock.Fifo())
    async def run():
        await flock.barrier().wait()
        return flock.get_rank()

    assert run() == [0, 1, 2]


def test_distribute_rejects_bad_policy():
    with pytest.raises(FlockUsageError, match="policy must be"):

        @flock.distribute(workers=2, policy="fifo")
        async def run():
            return flock.get_rank()


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
        flock.get_rank()


def test_per_rank_shards_positional_argument():
    world = 4

    @flock.distribute(workers=world)
    async def run(data: int):
        return data

    assert run(flock.per_rank([10, 20, 30, 40])) == [10, 20, 30, 40]


def test_per_rank_mixed_with_broadcast_argument():
    world = 3

    @flock.distribute(workers=world)
    async def run(data: int, scale: int):
        return data * scale

    assert run(flock.per_rank([1, 2, 3]), scale=10) == [10, 20, 30]


def test_per_rank_shards_keyword_argument():
    world = 2

    @flock.distribute(workers=world)
    async def run(*, data: str):
        return data

    assert run(data=flock.per_rank(["a", "b"])) == ["a", "b"]


def test_per_rank_rejects_wrong_length():
    world = 4

    @flock.distribute(workers=world)
    async def run(data: int):
        return data

    with pytest.raises(FlockUsageError, match="per_rank"):
        run(flock.per_rank([1, 2, 3]))


def test_plain_list_is_broadcast_to_all_ranks():
    world = 3

    @flock.distribute(workers=world)
    async def run(data: list[int]):
        return data

    shared = [1, 2, 3]
    assert run(shared) == [shared, shared, shared]
