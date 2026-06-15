from typing import Any

from flock.p2p.ops import Irecv, Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.work import Work


def isend[T](dst: Rank, value: T) -> Work[None]:
    runtime = require_runtime()
    return Work(runtime.begin_p2p(Isend(dst, value)), runtime)


def irecv(src: Rank) -> Work[Any]:
    runtime = require_runtime()
    return Work(runtime.begin_p2p(Irecv(src)), runtime)


async def send[T](dst: Rank, value: T) -> None:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Send(dst, value))
    await Work(handle, runtime).wait()


async def recv(src: Rank) -> Any:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Recv(src))
    return await Work(handle, runtime).wait()
