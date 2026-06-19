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
    def __init__(self, handle: Handle, expected_type: type[T] | None) -> None:
        self._handle = handle
        self._expected_type = expected_type

    def __await__(self) -> Generator[Handle, Any, T]:
        value = yield self._handle
        if self._expected_type is not None and not isinstance(value, self._expected_type):
            msg = f"expected {self._expected_type.__name__}, got {type(value).__name__}"
            raise TypeError(msg)
        return value


class Work[T]:
    def __init__(
        self,
        handle: Handle,
        runtime: Runtime,
        *,
        expected_type: type[T] | None = None,
    ) -> None:
        self._handle = handle
        self._runtime = runtime
        self._expected_type = expected_type

    def wait(self) -> _Wait[T]:
        return _Wait(self._handle, self._expected_type)

    def is_completed(self) -> bool:
        return self._runtime.is_complete(self._handle)

    def __await__(self) -> None:
        raise FlockUsageError("await work.wait(), not await work")
