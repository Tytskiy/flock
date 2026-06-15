import flock
from flock import new_group


def test_gather_to_dst():
    @flock.distribute(workers=3)
    async def run():
        return await flock.gather(flock.get_rank() * 10, dst=0).wait()

    assert run() == [[0, 10, 20], None, None]


def test_gather_subgroup_ordering():
    group = new_group([2, 0])

    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        if rank not in group:
            return None
        return await flock.gather(rank * 10, dst=2, group=group).wait()

    assert run() == [None, None, [0, 20], None]
