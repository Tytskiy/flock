from collections.abc import Generator
from typing import Any

from flock.collectives.handle import CollectiveHandle
from flock.errors import FlockUsageError
from flock.p2p.handle import P2PHandle

Handle = P2PHandle | CollectiveHandle


class Request:
    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        self._awaited = False

    def __await__(self) -> Generator[Handle, Any, Any]:
        self._awaited = True
        return (yield self._handle)

    def __del__(self) -> None:
        if not self._awaited:
            raise FlockUsageError(
                f"flock.{self._handle.kind}(...) was created but never awaited",
                RuntimeWarning,
                stacklevel=2,
            )
