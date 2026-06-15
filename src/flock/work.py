from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from flock.collectives.handle import CollectiveHandle
from flock.errors import FlockUsageError
from flock.p2p.handle import P2PHandle

if TYPE_CHECKING:
    from flock.scheduler.runtime import Runtime

Handle = P2PHandle | CollectiveHandle


class _Wait[T]:
    def __init__(self, handle: Handle) -> None:
        self._handle = handle

    def __await__(self) -> Generator[Handle, Any, T]:
        return (yield self._handle)


class Work[T]:
    def __init__(self, handle: Handle, runtime: Runtime) -> None:
        self._handle = handle
        self._runtime = runtime

    def wait(self) -> _Wait[T]:
        return _Wait(self._handle)

    def is_completed(self) -> bool:
        return self._runtime.is_complete(self._handle)

    def __await__(self) -> None:
        raise FlockUsageError("await work.wait(), not await work")
