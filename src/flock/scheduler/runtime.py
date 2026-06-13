import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from flock.collectives.engine import CollectiveEngine
from flock.collectives.handle import CollectiveHandle
from flock.errors import FlockUsageError
from flock.p2p.engine import P2PEngine
from flock.p2p.handle import P2PHandle
from flock.types import Rank


@dataclass(frozen=True)
class Runtime:
    p2p: P2PEngine
    collectives: CollectiveEngine

    def wait(self, rank: Rank, handle: object) -> None:
        match handle:
            case P2PHandle() as p2p_handle:
                self.p2p.wait(rank, p2p_handle)
            case CollectiveHandle() as collective_handle:
                self.collectives.wait(rank, collective_handle)
            case _:
                raise TypeError(f"unknown handle: {handle!r}")


_active: contextvars.ContextVar[Runtime | None] = contextvars.ContextVar(
    "flock_active_runtime",
    default=None,
)


def require_runtime() -> Runtime:
    runtime = _active.get()
    if runtime is None:
        raise FlockUsageError(
            "flock operations were used in the wrong place.\n"
            "They only work inside a function you start with @flock.distribute:\n\n"
            "    @flock.distribute(workers=4)\n"
            "    async def run():\n"
            "        await flock.send(1, msg)"
        )
    return runtime


@contextmanager
def active_runtime(runtime: Runtime) -> Iterator[None]:
    token = _active.set(runtime)
    try:
        yield
    finally:
        _active.reset(token)
