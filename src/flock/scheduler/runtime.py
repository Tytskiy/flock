import contextvars
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from flock.collectives.engine import CollectiveEngine
from flock.collectives.handle import CollectiveHandle
from flock.collectives.ops import CollectiveCall
from flock.errors import FlockUsageError
from flock.p2p.engine import P2PEngine
from flock.p2p.handle import P2PHandle
from flock.p2p.ops import P2PCall
from flock.types import Group, Rank

Handle = P2PHandle | CollectiveHandle


class PendingRegistry:
    def __init__(self) -> None:
        self._pending: defaultdict[Rank, list[Handle]] = defaultdict(list)

    def register(self, handle: Handle) -> None:
        self._pending[handle.rank].append(handle)

    def mark_awaited(self, handle: Handle) -> None:
        pending = self._pending[handle.rank]
        try:
            pending.remove(handle)
        except ValueError:
            raise FlockUsageError(
                f"rank {handle.rank} tried to wait on flock.{handle.kind}(...) again.\n"
                "Each Work can only be awaited once."
            ) from None

    def check(self, rank: Rank) -> None:
        lines = self._bullet_lines_for(rank)
        if not lines:
            return
        raise FlockUsageError(
            f"rank {rank} finished with unawaited operations:\n{lines}\n"
            "Every flock operation must be awaited before the rank returns."
        )

    def report_lines(self) -> list[str]:
        lines: list[str] = []
        for rank in sorted(rank for rank, pending in self._pending.items() if pending):
            for operation in self._operations_for(rank):
                lines.append(f"rank {rank}: {operation}")
        return lines

    def _bullet_lines_for(self, rank: Rank) -> str:
        operations = self._operations_for(rank)
        if not operations:
            return ""
        return "\n".join(f"  - {operation}" for operation in operations)

    def _operations_for(self, rank: Rank) -> list[str]:
        return [f"flock.{handle.kind}(...)" for handle in self._pending[rank]]


@dataclass(frozen=True)
class Runtime:
    p2p: P2PEngine
    collectives: CollectiveEngine
    pending: PendingRegistry = field(default_factory=PendingRegistry)

    def begin_p2p(self, call: P2PCall) -> P2PHandle:
        handle = self.p2p.begin(call)
        self.pending.register(handle)
        return handle

    def begin_collective(self, group: Group, call: CollectiveCall) -> CollectiveHandle:
        handle = self.collectives.begin(group, call)
        self.pending.register(handle)
        return handle

    def wait(self, rank: Rank, handle: object) -> None:
        match handle:
            case P2PHandle() as p2p_handle:
                self.pending.mark_awaited(p2p_handle)
                self.p2p.wait(rank, p2p_handle)
            case CollectiveHandle() as collective_handle:
                self.pending.mark_awaited(collective_handle)
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
