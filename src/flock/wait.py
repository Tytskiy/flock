from collections.abc import Generator
from typing import Any

from flock.collectives.handle import CollectiveHandle
from flock.p2p.handle import P2PHandle

Handle = P2PHandle | CollectiveHandle


class Request:
    def __init__(self, handle: Handle) -> None:
        self._handle = handle

    def __await__(self) -> Generator[Handle, Any, Any]:
        return (yield self._handle)
