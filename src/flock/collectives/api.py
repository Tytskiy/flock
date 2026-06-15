from collections.abc import Sequence
from typing import Any

from flock.collectives.ops import (
    AllGather,
    AllReduce,
    AllToAll,
    Barrier,
    Broadcast,
    Gather,
    Reduce,
    ReduceFn,
    ReduceOp,
    ReduceOpLike,
    ReduceScatter,
    Scatter,
)
from flock.scheduler.runtime import require_runtime
from flock.types import WORLD, Group, Rank
from flock.work import Work


def _normalize_op(op: ReduceOp | str | ReduceFn) -> ReduceOpLike:
    return op if callable(op) else ReduceOp(op)


def barrier(group: Group = WORLD) -> Work[None]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Barrier()), runtime)


def all_gather[T](value: T, group: Group = WORLD) -> Work[list[T]]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, AllGather(value=value)), runtime)


def all_reduce[T](value: T, op: ReduceOp | str | ReduceFn, group: Group = WORLD) -> Work[T]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, AllReduce(value=value, op=_normalize_op(op))), runtime)


def reduce[T](
    value: T,
    op: ReduceOp | str | ReduceFn,
    dst: Rank = 0,
    group: Group = WORLD,
) -> Work[T | None]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Reduce(value=value, op=_normalize_op(op), dst=dst)), runtime)


def broadcast[T](value: T | None, src: Rank = 0, group: Group = WORLD) -> Work[T]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Broadcast(value=value, src=src)), runtime)


def gather[T](value: T, dst: Rank = 0, group: Group = WORLD) -> Work[list[T] | None]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Gather(value=value, dst=dst)), runtime)


def reduce_scatter[T](
    values: Sequence[T],
    op: ReduceOp | str | ReduceFn,
    group: Group = WORLD,
) -> Work[T]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, ReduceScatter(values=values, op=_normalize_op(op))), runtime)


def all_to_all[T](values: Sequence[T], group: Group = WORLD) -> Work[list[T]]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, AllToAll(values=values)), runtime)


def scatter(
    values: Sequence[Any] | None,
    src: Rank = 0,
    group: Group = WORLD,
) -> Work[Any]:
    runtime = require_runtime()
    return Work(runtime.begin_collective(group, Scatter(values=values, src=src)), runtime)
