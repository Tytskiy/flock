import flock
from flock import new_group


def test_reduce_to_dst():
    @flock.distribute(workers=4)
    async def run():
        return await flock.reduce(flock.get_rank(), "sum", dst=1).wait()

    assert run() == [None, 6, None, None]


def test_reduce_subgroup():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        group = await new_group([0, 2, 3])
        if rank not in group:
            return None
        return await flock.reduce(rank, "sum", dst=3, group=group).wait()

    assert run() == [None, None, None, 5]
