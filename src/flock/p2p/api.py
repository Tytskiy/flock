from typing import Any

from flock.p2p.ops import Irecv, Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.work import Work


def isend[T](dst: Rank, value: T) -> Work[None]:
    return Work(require_runtime().begin_p2p(Isend(dst, value)))


def irecv(src: Rank) -> Work[Any]:
    return Work(require_runtime().begin_p2p(Irecv(src)))


async def send[T](dst: Rank, value: T) -> None:
    handle = require_runtime().begin_p2p(Send(dst, value))
    await Work(handle).wait()


async def recv(src: Rank) -> Any:
    handle = require_runtime().begin_p2p(Recv(src))
    return await Work(handle).wait()
