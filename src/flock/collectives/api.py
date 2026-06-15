from collections.abc import Sequence
from typing import Any

from flock.collectives.ops import AllGather, AllReduce, Barrier, ReduceOp, Scatter
from flock.scheduler.runtime import require_runtime
from flock.types import WORLD, Group, Rank
from flock.work import Work


def barrier(group: Group = WORLD) -> Work[None]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Barrier()), runtime)


def all_gather[T](value: T, group: Group = WORLD) -> Work[list[T]]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, AllGather(value=value)), runtime)


def all_reduce[T](value: T, op: ReduceOp | str, group: Group = WORLD) -> Work[T]:
    op = ReduceOp(op)
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, AllReduce(value=value, op=op)), runtime)


def scatter(
    values: Sequence[Any] | None,
    src: Rank = 0,
    group: Group = WORLD,
) -> Work[Any]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Scatter(values=values, src=src)), runtime)
