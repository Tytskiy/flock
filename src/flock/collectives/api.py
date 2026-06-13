from collections.abc import Sequence

from flock.collectives.ops import AllGather, AllReduce, Barrier, ReduceOp, Scatter
from flock.scheduler.runtime import require_runtime
from flock.types import WORLD, Group, Rank
from flock.wait import Request


def barrier(group: Group | None = WORLD) -> Request:
    handle = require_runtime().collectives.begin(group, Barrier())
    return Request(handle)


def all_gather[T](value: T, group: Group | None = WORLD) -> Request:
    handle = require_runtime().collectives.begin(group, AllGather(value=value))
    return Request(handle)


def all_reduce[T](value: T, op: ReduceOp, group: Group | None = WORLD) -> Request:
    handle = require_runtime().collectives.begin(group, AllReduce(value=value, op=op))
    return Request(handle)


def scatter[T](
    values: Sequence[T] | None,
    root: Rank = 0,
    group: Group | None = WORLD,
) -> Request:
    handle = require_runtime().collectives.begin(group, Scatter(values=values, root=root))
    return Request(handle)
