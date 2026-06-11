from __future__ import annotations

import gc

import pytest

import flock
from flock import FlockDeadlockError, FlockUsageError


def test_rank_and_world_size():
    world = 3

    @flock.distribute(workers=world)
    async def run():
        return flock.rank(), flock.world_size()

    assert run() == [(0, 3), (1, 3), (2, 3)]


def test_isend_recv_ring():
    world = 4

    @flock.distribute(workers=world)
    async def run():
        rank = flock.rank()
        nxt = (rank + 1) % flock.world_size()
        prv = (rank - 1) % flock.world_size()
        await flock.isend(nxt, f"hi from {rank}")
        return await flock.recv(prv)

    assert run() == [f"hi from {(r - 1) % world}" for r in range(world)]


def test_synchronous_send():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            await flock.send(1, "ping")
            return "sent"
        return await flock.recv(0)

    assert run() == ["sent", "ping"]


def test_distribute_with_explicit_seed():
    world = 4

    @flock.distribute(workers=world, seed=123)
    async def run():
        rank = flock.rank()
        await flock.isend((rank + 1) % world, rank)
        return await flock.recv((rank - 1) % world)

    assert run() == [(r - 1) % world for r in range(world)]


def test_distribute_deadlock():
    @flock.distribute(workers=2)
    async def run():
        other = 1 - flock.rank()
        return await flock.recv(other)

    with pytest.raises(FlockDeadlockError):
        run()


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


def test_unawaited_op_warns():
    with pytest.warns(RuntimeWarning, match="never awaited"):
        flock.send(0, "x")
        gc.collect()


def test_awaited_op_does_not_warn():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            await flock.isend(1, "x")
            return "sent"
        return await flock.recv(0)

    assert run() == ["sent", "x"]
