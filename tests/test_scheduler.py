from __future__ import annotations

import pytest

from flock import Fifo, FlockDeadlockError, ISendOp, Random, RecvOp, Scheduler, SendOp, Worker


class _Await:
    def __init__(self, op):
        self.op = op

    def __await__(self):
        return (yield self.op)


def run(coros, **kwargs):
    return Scheduler([Worker(coro) for coro in coros], **kwargs).run()


def test_isend_recv_ring():
    world = 4

    async def worker(rank):
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        await _Await(ISendOp(dst=nxt, value=f"hi from {rank}"))
        return await _Await(RecvOp(src=prv))

    results = run([worker(r) for r in range(world)])
    assert results == [f"hi from {(r - 1) % world}" for r in range(world)]


def test_synchronous_send_rendezvous():
    async def sender():
        await _Await(SendOp(dst=1, value="payload"))
        return "sent"

    async def receiver():
        return await _Await(RecvOp(src=0))

    results = run([sender(), receiver()])
    assert results == ["sent", "payload"]


def test_synchronous_send_to_waiting_receiver():
    async def receiver():
        return await _Await(RecvOp(src=1))

    async def sender():
        await _Await(SendOp(dst=0, value="hello"))
        return "done"

    results = run([receiver(), sender()])
    assert results == ["hello", "done"]


def test_synchronous_send_can_deadlock():
    async def worker(rank):
        other = 1 - rank
        await _Await(SendOp(dst=other, value=rank))
        return await _Await(RecvOp(src=other))

    with pytest.raises(FlockDeadlockError):
        run([worker(0), worker(1)])


def test_mutual_recv_deadlocks():
    async def worker(rank):
        other = 1 - rank
        return await _Await(RecvOp(src=other))

    with pytest.raises(FlockDeadlockError):
        run([worker(0), worker(1)])


@pytest.mark.parametrize("seed", range(8))
def test_results_are_deterministic_across_seeds(seed):
    world = 5

    async def worker(rank):
        await _Await(ISendOp(dst=(rank + 1) % world, value=rank))
        return await _Await(RecvOp(src=(rank - 1) % world))

    results = run([worker(r) for r in range(world)], policy=Random(seed=seed))
    assert results == [(r - 1) % world for r in range(world)]


@pytest.mark.parametrize("policy", [Random(seed=0), Fifo()])
def test_policies_agree_on_results(policy):
    world = 4

    async def worker(rank):
        await _Await(ISendOp(dst=(rank + 1) % world, value=rank))
        return await _Await(RecvOp(src=(rank - 1) % world))

    results = run([worker(r) for r in range(world)], policy=policy)
    assert results == [(r - 1) % world for r in range(world)]
