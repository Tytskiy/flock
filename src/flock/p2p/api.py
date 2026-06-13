from flock.p2p.ops import Isend, Recv, Send
from flock.scheduler.runtime import require_runtime
from flock.types import Rank
from flock.wait import Request


def isend[T](dst: Rank, value: T) -> Request:
    return Request(require_runtime().begin_p2p(Isend(dst, value)))


def send[T](dst: Rank, value: T) -> Request:
    return Request(require_runtime().begin_p2p(Send(dst, value)))


def recv[T](src: Rank) -> Request:
    return Request(require_runtime().begin_p2p(Recv(src)))
