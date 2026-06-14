from typing import Any, assert_type

import flock


async def _collectives() -> None:
    assert_type(flock.barrier(), flock.Work[None])
    assert_type(await flock.barrier().wait(), None)

    assert_type(flock.all_gather(1), flock.Work[list[int]])
    assert_type(await flock.all_gather(1).wait(), list[int])

    assert_type(flock.all_reduce(1.0, "sum"), flock.Work[float])
    assert_type(await flock.all_reduce(1.0, "sum").wait(), float)

    assert_type(flock.scatter(["a"], src=0), flock.Work[str])
    assert_type(await flock.scatter(["a"], src=0).wait(), str)


async def _recv_int(src: int) -> int:
    return await flock.recv(src)


async def _p2p() -> None:
    assert_type(flock.isend(0, 1), flock.Work[None])
    assert_type(await flock.isend(0, 1).wait(), None)

    assert_type(flock.irecv(0), flock.Work[Any])
    assert_type(await flock.irecv(0).wait(), Any)

    assert_type(await flock.send(0, "x"), None)
    assert_type(await _recv_int(0), int)
