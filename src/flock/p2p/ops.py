from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from flock.types import Rank


class P2PCall(Protocol):
    kind: ClassVar[str]

    @property
    def peer(self) -> Rank: ...

    @property
    def tag(self) -> int: ...


@dataclass(frozen=True)
class Isend:
    kind: ClassVar[str] = "isend"
    dst: Rank
    value: Any
    tag: int = 0

    @property
    def peer(self) -> Rank:
        return self.dst


@dataclass(frozen=True)
class Send:
    kind: ClassVar[str] = "send"
    dst: Rank
    value: Any
    tag: int = 0

    @property
    def peer(self) -> Rank:
        return self.dst


@dataclass(frozen=True)
class Recv:
    kind: ClassVar[str] = "recv"
    src: Rank
    tag: int = 0

    @property
    def peer(self) -> Rank:
        return self.src


@dataclass(frozen=True)
class Irecv:
    kind: ClassVar[str] = "irecv"
    src: Rank
    tag: int = 0

    @property
    def peer(self) -> Rank:
        return self.src
