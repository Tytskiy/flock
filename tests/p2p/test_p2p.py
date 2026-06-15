import pytest

import flock
from flock import FlockDeadlockError, FlockUsageError


def test_isend_recv_ring():
    world = 4

    @flock.distribute(workers=world)
    async def run():
        rank = flock.get_rank()
        nxt = (rank + 1) % flock.get_world_size()
        prv = (rank - 1) % flock.get_world_size()
        await flock.isend(nxt, f"hi from {rank}").wait()
        return await flock.recv(prv)

    assert run() == [f"hi from {(r - 1) % world}" for r in range(world)]


def test_synchronous_send():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            await flock.send(1, "ping")
            return "sent"
        return await flock.recv(0)

    assert run() == ["sent", "ping"]


def test_recv_deadlock():
    @flock.distribute(workers=2)
    async def run():
        other = 1 - flock.get_rank()
        return await flock.recv(other)

    with pytest.raises(FlockDeadlockError):
        run()


def test_unawaited_op_raises():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            flock.isend(1, "x")
        await flock.barrier().wait()
        return "done"

    with pytest.raises(FlockUsageError, match="unawaited"):
        run()


def test_exits_with_unawaited_send_raises():
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.get_rank() == 0:
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
        if flock.get_rank() == 0:
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


def test_recv_waits_for_expected_peer():
    @flock.distribute(workers=3, seed=0)
    async def run():
        if flock.get_rank() == 2:
            await flock.isend(1, "from-2").wait()
            return None
        if flock.get_rank() == 0:
            await flock.send(1, "from-0")
            return None
        return await flock.recv(0)

    assert run() == [None, "from-0", None]


def test_recv_picks_peer_from_out_of_order_mailbox():
    @flock.distribute(workers=3, seed=0)
    async def run():
        if flock.get_rank() == 2:
            await flock.isend(1, "from-2").wait()
            await flock.barrier().wait()
            await flock.barrier().wait()
            return None
        if flock.get_rank() == 0:
            await flock.barrier().wait()
            await flock.isend(1, "from-0").wait()
            await flock.barrier().wait()
            return None
        await flock.barrier().wait()
        await flock.barrier().wait()
        first = await flock.recv(0)
        second = await flock.recv(2)
        return (first, second)

    assert run()[1] == ("from-0", "from-2")


@pytest.mark.parametrize("peer", [-1, 2])
def test_p2p_invalid_peer_raises(peer):
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.get_rank() == 0:
            flock.isend(peer, "x")
            return None
        return await flock.recv(0)

    with pytest.raises(FlockUsageError, match="out of range"):
        run()


def test_isend_to_self():
    @flock.distribute(workers=1)
    async def run():
        await flock.isend(0, "self").wait()
        return await flock.recv(0)

    assert run() == ["self"]


def test_blocking_send_to_self_deadlocks():
    @flock.distribute(workers=1)
    async def run():
        await flock.send(0, "self")
        return await flock.recv(0)

    with pytest.raises(FlockDeadlockError):
        run()


def test_recv_selects_by_tag():
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.get_rank() == 0:
            await flock.isend(1, "a", tag=1).wait()
            await flock.isend(1, "b", tag=2).wait()
            return None
        second = await flock.recv(0, tag=2)
        first = await flock.recv(0, tag=1)
        return (first, second)

    assert run()[1] == ("a", "b")


def test_recv_any_tag_matches_any():
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.get_rank() == 0:
            await flock.isend(1, "x", tag=7).wait()
            return None
        return await flock.recv(0, tag=flock.ANY_TAG)

    assert run()[1] == "x"


def test_tag_mismatch_deadlocks():
    @flock.distribute(workers=2, seed=0)
    async def run():
        if flock.get_rank() == 0:
            await flock.isend(1, "x", tag=1).wait()
            return None
        return await flock.recv(0, tag=2)

    with pytest.raises(FlockDeadlockError):
        run()


def test_recv_any_source_gathers_all():
    world = 3

    @flock.distribute(workers=world, seed=0)
    async def run():
        rank = flock.get_rank()
        if rank != 0:
            await flock.isend(0, rank).wait()
            return None
        received = [await flock.recv(flock.ANY_SOURCE) for _ in range(world - 1)]
        return sorted(received)

    assert run()[0] == [1, 2]


def test_double_wait_raises():
    @flock.distribute(workers=2, seed=0)
    async def run():
        work = flock.isend(1, "x")
        await work.wait()
        await work.wait()

    with pytest.raises(FlockUsageError, match="again"):
        run()
