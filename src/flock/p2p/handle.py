from dataclasses import dataclass

from flock.types import Rank, RequestId


@dataclass(frozen=True)
class P2PHandle:
    kind: str
    rank: Rank
    peer: Rank
    request_id: RequestId
