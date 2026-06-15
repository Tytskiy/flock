from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from flock.errors import FlockCollectiveMismatch, FlockUsageError
from flock.types import Group, Rank


class ReduceOp(StrEnum):
    SUM = "sum"
    PROD = "prod"
    MIN = "min"
    MAX = "max"


ReduceFn = Callable[[Any, Any], Any]
ReduceOpLike = ReduceOp | ReduceFn


class CollectiveState(Protocol):
    kind: ClassVar[str]

    def complete(self, members: Group, rank: Rank) -> Any: ...


class CollectiveCall(Protocol):
    kind: ClassVar[str]

    def begin(self, rank: Rank) -> CollectiveState: ...

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None: ...


@dataclass(frozen=True)
class Barrier:
    kind: ClassVar[str] = "barrier"

    def begin(self, rank: Rank) -> BarrierState:
        return BarrierState()

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, BarrierState)


@dataclass(frozen=True)
class AllGather:
    kind: ClassVar[str] = "all_gather"
    value: Any

    def begin(self, rank: Rank) -> AllGatherState:
        return AllGatherState()

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, AllGatherState)
        state.values[rank] = self.value


@dataclass(frozen=True)
class AllReduce:
    kind: ClassVar[str] = "all_reduce"
    value: Any
    op: ReduceOpLike

    def begin(self, rank: Rank) -> AllReduceState:
        return AllReduceState(op=self.op)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, AllReduceState)
        _require_same_op("all_reduce", rank, self.op, state.op)
        state.value = reduce_value(self.op, state.value, self.value)


@dataclass(frozen=True)
class Reduce:
    kind: ClassVar[str] = "reduce"
    value: Any
    op: ReduceOpLike
    dst: Rank = 0

    def begin(self, rank: Rank) -> ReduceState:
        return ReduceState(op=self.op, dst=self.dst)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, ReduceState)
        _require_root("reduce", "dst", rank, self.dst, state.dst, members)
        _require_same_op("reduce", rank, self.op, state.op)
        state.value = reduce_value(self.op, state.value, self.value)


@dataclass(frozen=True)
class Broadcast:
    kind: ClassVar[str] = "broadcast"
    value: Any
    src: Rank = 0

    def begin(self, rank: Rank) -> BroadcastState:
        return BroadcastState(src=self.src)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, BroadcastState)
        _require_root("broadcast", "src", rank, self.src, state.src, members)
        if rank == self.src:
            state.value = self.value


@dataclass(frozen=True)
class Gather:
    kind: ClassVar[str] = "gather"
    value: Any
    dst: Rank = 0

    def begin(self, rank: Rank) -> GatherState:
        return GatherState(dst=self.dst)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, GatherState)
        _require_root("gather", "dst", rank, self.dst, state.dst, members)
        state.values[rank] = self.value


@dataclass(frozen=True)
class ReduceScatter:
    kind: ClassVar[str] = "reduce_scatter"
    values: Sequence[Any]
    op: ReduceOpLike

    def begin(self, rank: Rank) -> ReduceScatterState:
        return ReduceScatterState(op=self.op)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, ReduceScatterState)
        _require_same_op("reduce_scatter", rank, self.op, state.op)
        _require_one_value_per_member("reduce_scatter", rank, self.values, members)
        for position, value in enumerate(self.values):
            state.chunks[position] = reduce_value(self.op, state.chunks.get(position), value)


@dataclass(frozen=True)
class AllToAll:
    kind: ClassVar[str] = "all_to_all"
    values: Sequence[Any]

    def begin(self, rank: Rank) -> AllToAllState:
        return AllToAllState()

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, AllToAllState)
        _require_one_value_per_member("all_to_all", rank, self.values, members)
        state.inbox[rank] = list(self.values)


@dataclass(frozen=True)
class Scatter:
    kind: ClassVar[str] = "scatter"
    values: Sequence[Any] | None
    src: Rank = 0

    def begin(self, rank: Rank) -> ScatterState:
        return ScatterState(src=self.src)

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, ScatterState)
        _require_root("scatter", "src", rank, self.src, state.src, members)
        if rank == self.src:
            if self.values is None:
                raise FlockUsageError("scatter requires values on the src rank.")
            _require_one_value_per_member("scatter", rank, self.values, members)
            state.values = list(self.values)
        elif self.values is not None:
            raise FlockUsageError("only the scatter src rank should provide values.")


@dataclass
class BarrierState:
    kind: ClassVar[str] = "barrier"

    def complete(self, members: Group, rank: Rank) -> None:
        return None


@dataclass
class AllGatherState:
    kind: ClassVar[str] = "all_gather"
    values: dict[Rank, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any]:
        return copy.deepcopy([self.values[member] for member in members])


@dataclass
class AllReduceState:
    kind: ClassVar[str] = "all_reduce"
    op: ReduceOpLike
    value: Any | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.value)


@dataclass
class ReduceState:
    kind: ClassVar[str] = "reduce"
    op: ReduceOpLike
    dst: Rank
    value: Any | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        if rank != self.dst:
            return None
        return copy.deepcopy(self.value)


@dataclass
class BroadcastState:
    kind: ClassVar[str] = "broadcast"
    src: Rank
    value: Any = None

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.value)


@dataclass
class GatherState:
    kind: ClassVar[str] = "gather"
    dst: Rank
    values: dict[Rank, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any] | None:
        if rank != self.dst:
            return None
        return copy.deepcopy([self.values[member] for member in members])


@dataclass
class ReduceScatterState:
    kind: ClassVar[str] = "reduce_scatter"
    op: ReduceOpLike
    chunks: dict[int, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.chunks[members.index(rank)])


@dataclass
class AllToAllState:
    kind: ClassVar[str] = "all_to_all"
    inbox: dict[Rank, list[Any]] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any]:
        column = members.index(rank)
        return copy.deepcopy([self.inbox[sender][column] for sender in members])


@dataclass
class ScatterState:
    kind: ClassVar[str] = "scatter"
    src: Rank
    values: list[Any] | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        if self.values is None:
            raise FlockUsageError("scatter src did not provide values.")
        return copy.deepcopy(self.values[members.index(rank)])


_REDUCERS: dict[ReduceOp, ReduceFn] = {
    ReduceOp.SUM: operator.add,
    ReduceOp.PROD: operator.mul,
    ReduceOp.MIN: min,
    ReduceOp.MAX: max,
}


def _op_name(op: ReduceOpLike) -> str:
    return op.value if isinstance(op, ReduceOp) else getattr(op, "__name__", repr(op))


def _require_same_op(name: str, rank: Rank, mine: ReduceOpLike, theirs: ReduceOpLike) -> None:
    if isinstance(mine, ReduceOp) and mine != theirs:
        raise FlockCollectiveMismatch(
            f"rank {rank} called {name} with op {_op_name(mine)!r}, "
            f"but other ranks in the group already started with op {_op_name(theirs)!r}."
        )


def _require_root(kind: str, label: str, rank: Rank, root: Rank, established: Rank, members: Group) -> None:
    if root != established:
        raise FlockCollectiveMismatch(
            f"rank {rank} called {kind} with {label} {root}, "
            f"but other ranks in the group already started with {label} {established}."
        )
    if root not in members:
        raise FlockUsageError(f"{kind} {label} {root} is not in the group (members: {sorted(members)}).")


def _require_one_value_per_member(kind: str, rank: Rank, values: Sequence[Any], members: Group) -> None:
    if len(values) != len(members):
        raise FlockUsageError(
            f"{kind} expected {len(members)} values (one per member), but rank {rank} provided {len(values)}."
        )


def reduce_value(op: ReduceOpLike, acc: Any | None, curr: Any) -> Any:
    if acc is None:
        return curr
    reducer = _REDUCERS[op] if isinstance(op, ReduceOp) else op
    try:
        return reducer(acc, curr)
    except TypeError as exc:
        raise FlockUsageError(
            f"reduce could not combine {acc!r} and {curr!r} with op {_op_name(op)!r}; "
            "the values must support that reduction."
        ) from exc
