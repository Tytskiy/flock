import pytest

import flock
from flock.scheduler import Fifo


def test_isend_is_completed_immediately():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            work = flock.isend(1, "x")
            done = work.is_completed()
            await work.wait()
            return done
        return await flock.recv(0)

    assert run()[0] is True


def test_irecv_is_completed_transitions():
    @flock.distribute(workers=2)
    async def run():
        rank = flock.get_rank()
        if rank == 0:
            await flock.barrier().wait()
            await flock.isend(1, "hi").wait()
            await flock.barrier().wait()
            return None

        work = flock.irecv(0)
        before = work.is_completed()
        await flock.barrier().wait()
        await flock.barrier().wait()
        after = work.is_completed()
        value = await work.wait()
        return (before, after, value)

    assert run()[1] == (False, True, "hi")


def test_collective_is_completed_for_last_waiter(run_scheduler):
    polled: list[bool] = []

    async def worker(rank):
        work = flock.all_gather(rank)
        if rank == 1:
            polled.append(work.is_completed())
        return await work.wait()

    results = run_scheduler([worker(0), worker(1)], 2, policy=Fifo())
    assert polled == [True]
    assert results == [[0, 1], [0, 1]]


def test_irecv_expected_type_passes():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.send(1, "hi")
            return None

        return await flock.irecv(0, expected_type=str).wait()

    assert run()[1] == "hi"


def test_irecv_expected_type_rejects_mismatch():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.send(1, 1)
            return None

        return await flock.irecv(0, expected_type=str).wait()

    with pytest.raises(TypeError, match="expected str, got int"):
        run()


def test_recv_expected_type_passes():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.send(1, "hi")
            return None

        return await flock.recv(0, expected_type=str)

    assert run()[1] == "hi"


def test_recv_expected_type_rejects_mismatch():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.send(1, 1)
            return None

        return await flock.recv(0, expected_type=str)

    with pytest.raises(TypeError, match="expected str, got int"):
        run()
