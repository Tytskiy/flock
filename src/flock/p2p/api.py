from collections.abc import Coroutine
from typing import Any

from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.wait import Request


def isend[T](dst: Rank, value: T) -> Coroutine[Any, Any, None]:
    require_runtime()
    return _isend(dst, value)


async def _isend[T](dst: Rank, value: T) -> None:
    handle = require_runtime().p2p.begin_isend(dst, value)
    await Request(handle)


def send[T](dst: Rank, value: T) -> Coroutine[Any, Any, None]:
    require_runtime()
    return _send(dst, value)


async def _send[T](dst: Rank, value: T) -> None:
    handle = require_runtime().p2p.begin_send(dst, value)
    await Request(handle)


def recv[T](src: Rank) -> Coroutine[Any, Any, T]:
    require_runtime()
    return _recv(src)


async def _recv[T](src: Rank) -> T:
    handle = require_runtime().p2p.begin_recv(src)
    return await Request(handle)
