from dataclasses import dataclass

from flock.types import Group, Rank


@dataclass(frozen=True)
class CollectiveHandle:
    group: Group
    index: int
    rank: Rank
    kind: str
    request_id: int
