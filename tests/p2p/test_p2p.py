import gc

import pytest

import flock
from flock import FlockDeadlockError, FlockUsageError


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


def test_recv_deadlock():
    @flock.distribute(workers=2)
    async def run():
        other = 1 - flock.rank()
        return await flock.recv(other)

    with pytest.raises(FlockDeadlockError):
        run()


def test_unawaited_op_warns():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            flock.send(1, "x")
        await flock.barrier()
        return "done"

    with pytest.warns(RuntimeWarning, match="never awaited"):
        run()
        gc.collect()


def test_op_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.send(0, "x")


def test_awaited_op_does_not_warn():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            await flock.isend(1, "x")
            return "sent"
        return await flock.recv(0)

    assert run() == ["sent", "x"]
