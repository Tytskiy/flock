from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

Rank = int

RequestId = int

ANY_SOURCE: Rank = -1
ANY_TAG: int = -1

_WORLD_ID = 0


@dataclass(frozen=True, eq=False)
class Group:
    ranks: tuple[Rank, ...]
    id: int

    def __iter__(self) -> Iterator[Rank]:
        return iter(self.ranks)

    def __len__(self) -> int:
        return len(self.ranks)

    def __contains__(self, rank: object) -> bool:
        return rank in self.ranks

    def index(self, rank: Rank) -> int:
        return self.ranks.index(rank)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Group):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __lt__(self, other: Group) -> bool:
        return self.id < other.id


WORLD = Group(ranks=(), id=_WORLD_ID)


def _default_world_group(world_size: int) -> Group:
    return Group(ranks=tuple(range(world_size)), id=_WORLD_ID)
