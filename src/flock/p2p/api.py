from typing import Any, overload

from flock.p2p.ops import Irecv, Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.work import Work


def isend[T](dst: Rank, value: T, tag: int = 0) -> Work[None]:
    runtime = require_runtime()
    return Work(runtime.begin_p2p(Isend(dst, value, tag)), runtime)


@overload
def irecv(src: Rank, tag: int = 0) -> Work[Any]: ...


@overload
def irecv[T](src: Rank, tag: int = 0, *, expected_type: type[T]) -> Work[T]: ...


def irecv(
    src: Rank,
    tag: int = 0,
    *,
    expected_type: type[Any] | None = None,
) -> Work[Any]:
    runtime = require_runtime()
    return Work(
        runtime.begin_p2p(Irecv(src, tag)),
        runtime,
        expected_type=expected_type,
    )


async def send[T](dst: Rank, value: T, tag: int = 0) -> None:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Send(dst, value, tag))
    await Work(handle, runtime).wait()


@overload
async def recv(src: Rank, tag: int = 0) -> Any: ...


@overload
async def recv[T](src: Rank, tag: int = 0, *, expected_type: type[T]) -> T: ...


async def recv(
    src: Rank,
    tag: int = 0,
    *,
    expected_type: type[Any] | None = None,
) -> Any:
    runtime = require_runtime()
    handle = runtime.begin_p2p(Recv(src, tag))
    return await Work(handle, runtime, expected_type=expected_type).wait()
