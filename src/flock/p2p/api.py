from typing import Any, cast

from flock.p2p.ops import Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.wait import Work


def isend[T](dst: Rank, value: T) -> Work[None]:
    return Work(require_runtime().begin_p2p(Isend(dst, value)))


def irecv(src: Rank) -> Work[Any]:
    return Work(require_runtime().begin_p2p(Recv(src)))


async def send[T](dst: Rank, value: T) -> None:
    handle = require_runtime().begin_p2p(Send(dst, value))
    await Work(handle).wait()


async def recv[T](src: Rank) -> T:
    return cast(T, await irecv(src).wait())
