import pytest

import flock
from flock import FlockDeadlockError, FlockUsageError, new_group


def test_barrier_sync():
    @flock.distribute(workers=4)
    async def run():
        await flock.barrier().wait()
        return flock.get_rank()

    assert run() == [0, 1, 2, 3]


def test_barrier_async():
    @flock.distribute(workers=3)
    async def run():
        future = flock.barrier()
        await flock.isend((flock.get_rank() + 1) % flock.get_world_size(), flock.get_rank()).wait()
        await future.wait()
        return flock.get_rank()

    assert run() == [0, 1, 2]


def test_barrier_subgroup():
    group = new_group([0, 2])

    @flock.distribute(workers=4)
    async def run():
        if flock.get_rank() in group:
            await flock.barrier(group=group).wait()
            return "synced"
        return "skipped"

    assert run() == ["synced", "skipped", "synced", "skipped"]


def test_barrier_world_constant():
    @flock.distribute(workers=2)
    async def run():
        await flock.barrier(group=flock.WORLD).wait()
        return "ok"

    assert run() == ["ok", "ok"]


def test_barrier_deadlock():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.barrier().wait()
        return "done"

    with pytest.raises(FlockDeadlockError, match="barrier"):
        run()


def test_barrier_outside_distribute_raises():
    with pytest.raises(FlockUsageError, match="wrong place"):
        flock.barrier()


def test_barrier_non_member_raises():
    @flock.distribute(workers=3)
    async def run():
        if flock.get_rank() == 2:
            return flock.barrier(group=new_group([0, 1]))
        await flock.barrier().wait()
        return flock.get_rank()

    with pytest.raises(FlockUsageError, match="not a member"):
        run()


def test_unawaited_barrier_raises():
    @flock.distribute(workers=2)
    async def run():
        flock.barrier()
        await flock.barrier().wait()
        return flock.get_rank()

    with pytest.raises(FlockUsageError, match="unawaited"):
        run()
