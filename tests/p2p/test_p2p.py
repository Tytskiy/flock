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
        await flock.isend(nxt, f"hi from {rank}").wait()
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


def test_unawaited_op_raises():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            flock.isend(1, "x")
        await flock.barrier().wait()
        return "done"

    with pytest.raises(FlockUsageError, match="unawaited"):
        run()


def test_exits_with_unawaited_send_raises():
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.rank() == 0:
            flock.isend(1, "ping")
            return None
        return await flock.recv(0)

    with pytest.raises(FlockUsageError, match="flock.isend"):
        run()


def test_op_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.isend(0, "x")


def test_awaited_op_does_not_warn():
    @flock.distribute(workers=2)
    async def run():
        if flock.rank() == 0:
            await flock.isend(1, "x").wait()
            return "sent"
        return await flock.recv(0)

    assert run() == ["sent", "x"]


def test_await_work_directly_raises():
    @flock.distribute(workers=2)
    async def run():
        work = flock.isend(1, "x")
        await work

    with pytest.raises(FlockUsageError, match="await work.wait"):
        run()
