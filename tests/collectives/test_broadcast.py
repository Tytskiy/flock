import flock
from flock import new_group


def test_broadcast():
    @flock.distribute(workers=4)
    async def run():
        value = "payload" if flock.get_rank() == 2 else None
        return await flock.broadcast(value, src=2).wait()

    assert run() == ["payload"] * 4


def test_broadcast_returns_independent_copies():
    @flock.distribute(workers=2)
    async def run():
        rank = flock.get_rank()
        got = await flock.broadcast([1, 2, 3] if rank == 0 else None, src=0).wait()
        got.append(rank)
        return got

    assert run() == [[1, 2, 3, 0], [1, 2, 3, 1]]


def test_broadcast_subgroup():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        group = await new_group([1, 3])
        if rank not in group:
            return None
        value = "hi" if rank == 1 else None
        return await flock.broadcast(value, src=1, group=group).wait()

    assert run() == [None, "hi", None, "hi"]
