from collections.abc import Coroutine
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Worker[R]:
    coro: Coroutine[Any, Any, R]
    context: Context = field(default_factory=copy_context)


class Policy:
    pass


@dataclass(frozen=True)
class Random(Policy):
    seed: int | None = None


@dataclass(frozen=True)
class Fifo(Policy):
    pass


class Scheduler[R](Protocol):
    def run(self) -> list[R]: ...
