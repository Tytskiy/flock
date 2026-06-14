from collections.abc import Sequence
from typing import Any

from flock.collectives.ops import AllGather, AllReduce, Barrier, ReduceOp, Scatter
from flock.scheduler.runtime import require_runtime
from flock.types import WORLD, Group, Rank
from flock.work import Work


def barrier(group: Group | None = WORLD) -> Work[None]:
    return Work(require_runtime().begin_collective(group, Barrier()))


def all_gather[T](value: T, group: Group | None = WORLD) -> Work[list[T]]:
    return Work(require_runtime().begin_collective(group, AllGather(value=value)))


def all_reduce[T](value: T, op: ReduceOp | str, group: Group | None = WORLD) -> Work[T]:
    op = ReduceOp(op)
    return Work(require_runtime().begin_collective(group, AllReduce(value=value, op=op)))


def scatter(
    values: Sequence[Any] | None,
    src: Rank = 0,
    group: Group | None = WORLD,
) -> Work[Any]:
    return Work(require_runtime().begin_collective(group, Scatter(values=values, src=src)))
