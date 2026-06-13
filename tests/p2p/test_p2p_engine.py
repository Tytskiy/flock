import pytest

import flock


def test_isend_recv_ring(run_scheduler):
    world = 4

    async def worker(rank):
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        await flock.isend(nxt, f"hi from {rank}").wait()
        return await flock.recv(prv)

    results = run_scheduler([worker(r) for r in range(world)], world)
    assert results == [f"hi from {(r - 1) % world}" for r in range(world)]


def test_synchronous_send_rendezvous(run_scheduler):
    async def sender():
        await flock.send(1, "payload")
        return "sent"

    async def receiver():
        return await flock.recv(0)

    results = run_scheduler([sender(), receiver()], 2)
    assert results == ["sent", "payload"]


def test_synchronous_send_to_waiting_receiver(run_scheduler):
    async def receiver():
        return await flock.recv(1)

    async def sender():
        await flock.send(0, "hello")
        return "done"

    results = run_scheduler([receiver(), sender()], 2)
    assert results == ["hello", "done"]


def test_synchronous_send_can_deadlock(run_scheduler):
    async def worker(rank):
        other = 1 - rank
        await flock.send(other, rank)
        return await flock.recv(other)

    with pytest.raises(flock.FlockDeadlockError):
        run_scheduler([worker(0), worker(1)], 2)


def test_mutual_recv_deadlocks(run_scheduler):
    async def worker(rank):
        other = 1 - rank
        return await flock.recv(other)

    with pytest.raises(flock.FlockDeadlockError):
        run_scheduler([worker(0), worker(1)], 2)


def test_invalid_peer_raises(run_scheduler):
    async def worker():
        flock.isend(2, "x")

    with pytest.raises(flock.FlockUsageError, match="out of range"):
        run_scheduler([worker(), worker()], 2)
