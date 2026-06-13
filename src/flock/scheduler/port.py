from typing import Any, Protocol

from flock.types import Rank


class SchedulePort(Protocol):
    @property
    def world_size(self) -> int: ...

    def resume(self, rank: Rank, value: Any = None) -> None: ...
