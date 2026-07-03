from __future__ import annotations

import copy
import itertools
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from flock.errors import FlockCollectiveMismatch, FlockUsageError
from flock.payload import payload_bytes
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

    def payload_bytes(self) -> int: ...

    def rank_payload_bytes(self, rank: Rank) -> int: ...


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
        state.rank_bytes[rank] = payload_bytes(self.value)


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
        state.rank_bytes[rank] = payload_bytes(self.value)


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
            state.rank_bytes[rank] = payload_bytes(self.value)


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
        state.rank_bytes[rank] = sum(payload_bytes(value) for value in self.values)


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
        state.rank_bytes[rank] = sum(payload_bytes(value) for value in self.values)


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
            state.rank_bytes[rank] = sum(payload_bytes(value) for value in self.values)
        elif self.values is not None:
            raise FlockUsageError("only the scatter src rank should provide values.")


@dataclass(frozen=True)
class NewGroup:
    kind: ClassVar[str] = "new_group"
    ranks: Sequence[Rank]
    world_size: int

    def begin(self, rank: Rank) -> NewGroupState:
        ranks = normalize_group_ranks(self.ranks, self.world_size)
        return NewGroupState(ranks=ranks, group=Group(ranks=ranks, id=next(_group_ids)))

    def enter(self, state: CollectiveState, rank: Rank, members: Group) -> None:
        assert isinstance(state, NewGroupState)
        assert self.world_size == len(members)
        ranks = normalize_group_ranks(self.ranks, self.world_size)
        if ranks != state.ranks:
            raise FlockCollectiveMismatch(
                f"rank {rank} called new_group with members {list(ranks)}, "
                f"but other ranks in the group already started it with members {list(state.ranks)}."
            )


@dataclass
class BarrierState:
    kind: ClassVar[str] = "barrier"

    def complete(self, members: Group, rank: Rank) -> None:
        return None

    def payload_bytes(self) -> int:
        return 0

    def rank_payload_bytes(self, rank: Rank) -> int:
        return 0


@dataclass
class AllGatherState:
    kind: ClassVar[str] = "all_gather"
    values: dict[Rank, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any]:
        return copy.deepcopy([self.values[member] for member in members])

    def payload_bytes(self) -> int:
        return sum(payload_bytes(value) for value in self.values.values())

    def rank_payload_bytes(self, rank: Rank) -> int:
        if rank not in self.values:
            return 0
        return payload_bytes(self.values[rank])


@dataclass
class AllReduceState:
    kind: ClassVar[str] = "all_reduce"
    op: ReduceOpLike
    value: Any | None = None
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.value)

    def payload_bytes(self) -> int:
        return payload_bytes(self.value)

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class ReduceState:
    kind: ClassVar[str] = "reduce"
    op: ReduceOpLike
    dst: Rank
    value: Any | None = None
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        if rank != self.dst:
            return None
        return copy.deepcopy(self.value)

    def payload_bytes(self) -> int:
        return payload_bytes(self.value)

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class BroadcastState:
    kind: ClassVar[str] = "broadcast"
    src: Rank
    value: Any = None
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.value)

    def payload_bytes(self) -> int:
        return payload_bytes(self.value)

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class GatherState:
    kind: ClassVar[str] = "gather"
    dst: Rank
    values: dict[Rank, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any] | None:
        if rank != self.dst:
            return None
        return copy.deepcopy([self.values[member] for member in members])

    def payload_bytes(self) -> int:
        return sum(payload_bytes(value) for value in self.values.values())

    def rank_payload_bytes(self, rank: Rank) -> int:
        if rank not in self.values:
            return 0
        return payload_bytes(self.values[rank])


@dataclass
class ReduceScatterState:
    kind: ClassVar[str] = "reduce_scatter"
    op: ReduceOpLike
    chunks: dict[int, Any] = field(default_factory=dict)
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        return copy.deepcopy(self.chunks[members.index(rank)])

    def payload_bytes(self) -> int:
        return sum(payload_bytes(chunk) for chunk in self.chunks.values())

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class AllToAllState:
    kind: ClassVar[str] = "all_to_all"
    inbox: dict[Rank, list[Any]] = field(default_factory=dict)
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> list[Any]:
        column = members.index(rank)
        return copy.deepcopy([self.inbox[sender][column] for sender in members])

    def payload_bytes(self) -> int:
        return sum(payload_bytes(value) for values in self.inbox.values() for value in values)

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class ScatterState:
    kind: ClassVar[str] = "scatter"
    src: Rank
    values: list[Any] | None = None
    rank_bytes: dict[Rank, int] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        if self.values is None:
            raise FlockUsageError("scatter src did not provide values.")
        return copy.deepcopy(self.values[members.index(rank)])

    def payload_bytes(self) -> int:
        if self.values is None:
            return 0
        return sum(payload_bytes(value) for value in self.values)

    def rank_payload_bytes(self, rank: Rank) -> int:
        return self.rank_bytes.get(rank, 0)


@dataclass
class NewGroupState:
    kind: ClassVar[str] = "new_group"
    ranks: tuple[Rank, ...]
    group: Group

    def complete(self, members: Group, rank: Rank) -> Group:
        return self.group

    def payload_bytes(self) -> int:
        return 0

    def rank_payload_bytes(self, rank: Rank) -> int:
        return 0


_REDUCERS: dict[ReduceOp, ReduceFn] = {
    ReduceOp.SUM: operator.add,
    ReduceOp.PROD: operator.mul,
    ReduceOp.MIN: min,
    ReduceOp.MAX: max,
}

_group_ids = itertools.count(1)


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


def normalize_group_ranks(ranks: Sequence[Rank], world_size: int) -> tuple[Rank, ...]:
    unique = sorted(set(ranks))
    if len(unique) != len(ranks):
        raise FlockUsageError(f"new_group got duplicate ranks: {sorted(ranks)}.")
    if not unique:
        raise FlockUsageError("new_group requires at least one rank.")
    for member in unique:
        if member < 0 or member >= world_size:
            raise FlockUsageError(f"group member {member} is out of range for world_size={world_size}.")
    return tuple(unique)


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
