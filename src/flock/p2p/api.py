from typing import Any

from flock.p2p.ops import Irecv, Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.work import Work


def isend[T](dst: Rank, value: T, tag: int = 0) -> Work[None]:
    runtime = require_runtime()
    return Work(runtime.begin_p2p(Isend(dst, value, tag)), runtime)


def irecv(src: Rank, tag: int = 0) -> Work[Any]:
    runtime = require_runtime()
    return Work(runtime.begin_p2p(Irecv(src, tag)), runtime)


async def send[T](dst: Rank, value: T, tag: int = 0) -> None:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Send(dst, value, tag))
    await Work(handle, runtime).wait()


async def recv(src: Rank, tag: int = 0) -> Any:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Recv(src, tag))
    return await Work(handle, runtime).wait()
