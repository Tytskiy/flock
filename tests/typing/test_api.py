from typing import Any, assert_type

import flock


async def _collectives() -> None:
    assert_type(flock.barrier(), flock.Work[None])
    assert_type(await flock.barrier().wait(), None)

    assert_type(flock.all_gather(1), flock.Work[list[int]])
    assert_type(await flock.all_gather(1).wait(), list[int])

    assert_type(flock.all_reduce(1.0, "sum"), flock.Work[float])
    assert_type(await flock.all_reduce(1.0, "sum").wait(), float)

    assert_type(flock.scatter(["a"], src=0), flock.Work[Any])
    assert_type(await flock.scatter(["a"], src=0).wait(), Any)

    assert_type(flock.broadcast(1, src=0), flock.Work[int])
    assert_type(await flock.broadcast(1, src=0).wait(), int)

    assert_type(flock.reduce(1.0, "sum", dst=0), flock.Work[float | None])
    assert_type(await flock.reduce(1.0, "sum", dst=0).wait(), float | None)

    assert_type(flock.gather(1, dst=0), flock.Work[list[int] | None])
    assert_type(await flock.gather(1, dst=0).wait(), list[int] | None)

    assert_type(flock.reduce_scatter([1, 2], "sum"), flock.Work[int])
    assert_type(await flock.reduce_scatter([1, 2], "sum").wait(), int)

    assert_type(flock.all_to_all([1, 2]), flock.Work[list[int]])
    assert_type(await flock.all_to_all([1, 2]).wait(), list[int])

    assert_type(flock.all_reduce(1, lambda a, b: a + b), flock.Work[int])
    assert_type(await flock.new_group([0, 1]), flock.Group)


async def _recv_int(src: int) -> int:
    return await flock.recv(src)


async def _p2p() -> None:
    assert_type(flock.isend(0, 1), flock.Work[None])
    assert_type(await flock.isend(0, 1).wait(), None)

    assert_type(flock.irecv(0), flock.Work[Any])
    assert_type(await flock.irecv(0).wait(), Any)
    assert_type(flock.irecv(0, expected_type=str), flock.Work[str])
    assert_type(await flock.irecv(0, expected_type=str).wait(), str)

    assert_type(await flock.send(0, "x"), None)
    assert_type(await flock.recv(0, expected_type=str), str)
    assert_type(await _recv_int(0), int)


def _distributed_with_per_rank(values: list[int]) -> list[int]:
    @flock.distribute(workers=2)
    async def run(data: int) -> int:
        return data

    return run(flock.per_rank(values))


def _per_rank_types() -> None:
    assert_type(flock.per_rank([1, 2]), flock.PerRank[int])
    assert_type(_distributed_with_per_rank([1, 2]), list[int])
