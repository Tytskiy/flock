import contextvars
from dataclasses import dataclass

from flock.errors import FlockUsageError
from flock.types import Rank


@dataclass(frozen=True)
class Context:
    rank: Rank
    world_size: int


_current: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "flock_current_context", default=None
)


def make_context(rank: Rank, world_size: int) -> contextvars.Context:
    ctx = contextvars.copy_context()
    ctx.run(_current.set, Context(rank=rank, world_size=world_size))
    return ctx


def current() -> Context:
    ctx = _current.get()
    if ctx is None:
        raise FlockUsageError(
            "flock.rank() or flock.world_size() were used in the wrong place.\n"
            "They only work inside a function you start with @flock.distribute:\n\n"
            "    @flock.distribute(workers=4)\n"
            "    async def run():\n"
            "        print(flock.rank(), flock.world_size())"
        )
    return ctx


def rank() -> Rank:
    return current().rank


def world_size() -> int:
    return current().world_size
