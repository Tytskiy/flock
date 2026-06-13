from dataclasses import dataclass

from flock.types import Rank


@dataclass(frozen=True)
class P2PHandle:
    kind: str
    rank: Rank
    peer: Rank
    request_id: int
