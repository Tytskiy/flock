import pytest

import flock
from flock import FlockCollectiveMismatch
from flock.collectives.ops import AllReduce, Barrier
from flock.context import make_context
from flock.scheduler import CooperativeScheduler, Fifo, Worker
from flock.types import WORLD, Rank


def _register(
    scheduler: CooperativeScheduler[object],
    rank: Rank,
    collective: Barrier | AllReduce,
) -> None:
    worker = scheduler.workers[rank]
    worker.context.run(scheduler.runtime.collectives.begin, WORLD, collective)


def test_collective_kind_mismatch():
    async def idle():
        return None

    scheduler = CooperativeScheduler(
        [
            Worker(idle(), context=make_context(0, 2)),
            Worker(idle(), context=make_context(1, 2)),
        ],
        policy=Fifo(),
    )

    try:
        _register(scheduler, 0, Barrier())
        with pytest.raises(FlockCollectiveMismatch, match="all_reduce"):
            _register(scheduler, 1, AllReduce(value=0, op="sum"))
    finally:
        for worker in scheduler.workers:
            worker.coro.close()


def test_barrier_scheduler_low_level():
    async def participant():
        await flock.barrier().wait()
        return "done"

    results = CooperativeScheduler(
        [
            Worker(participant(), context=make_context(0, 2)),
            Worker(participant(), context=make_context(1, 2)),
        ],
        policy=Fifo(),
    ).run()
    assert results == ["done", "done"]
