from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from flock.errors import FlockUsageError

Rank = int

RequestId = int

_WORLD_ID = 0
_group_ids = itertools.count(_WORLD_ID + 1)


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


WORLD = Group(ranks=(), id=_WORLD_ID)


def new_group(ranks: Sequence[Rank]) -> Group:
    unique = sorted(set(ranks))
    if len(unique) != len(ranks):
        raise FlockUsageError(f"new_group got duplicate ranks: {sorted(ranks)}.")
    return Group(ranks=tuple(unique), id=next(_group_ids))


def _default_world_group(world_size: int) -> Group:
    return Group(ranks=tuple(range(world_size)), id=_WORLD_ID)
