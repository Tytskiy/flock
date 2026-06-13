from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from flock.errors import FlockCollectiveMismatch, FlockUsageError
from flock.types import Group, Rank

ReduceOp = Literal["sum", "prod", "min", "max"]


class CollectiveState(Protocol):
    kind: ClassVar[str]

    def complete(self, members: Group, rank: Rank) -> Any: ...


class CollectiveCall(Protocol):
    kind: ClassVar[str]

    def begin(self, rank: Rank) -> CollectiveState: ...

    def enter(self, state: CollectiveState, rank: Rank) -> None: ...


@dataclass(frozen=True)
class Barrier:
    kind: ClassVar[str] = "barrier"

    def begin(self, rank: Rank) -> BarrierState:
        return BarrierState()

    def enter(self, state: CollectiveState, rank: Rank) -> None:
        if not isinstance(state, BarrierState):
            raise TypeError(f"expected BarrierState, got {type(state).__name__}")


@dataclass(frozen=True)
class AllGather:
    kind: ClassVar[str] = "all_gather"
    value: Any

    def begin(self, rank: Rank) -> AllGatherState:
        return AllGatherState()

    def enter(self, state: CollectiveState, rank: Rank) -> None:
        if not isinstance(state, AllGatherState):
            raise TypeError(f"expected AllGatherState, got {type(state).__name__}")
        state.values[rank] = self.value


@dataclass(frozen=True)
class AllReduce:
    kind: ClassVar[str] = "all_reduce"
    value: Any
    op: ReduceOp

    def begin(self, rank: Rank) -> AllReduceState:
        return AllReduceState(op=self.op)

    def enter(self, state: CollectiveState, rank: Rank) -> None:
        if not isinstance(state, AllReduceState):
            raise TypeError(f"expected AllReduceState, got {type(state).__name__}")
        if state.op != self.op:
            raise FlockCollectiveMismatch(
                f"rank {rank} called all_reduce with op {self.op!r}, "
                f"but other ranks in the group already started with op {state.op!r}."
            )
        state.values[rank] = self.value


@dataclass(frozen=True)
class Scatter:
    kind: ClassVar[str] = "scatter"
    values: Sequence[Any] | None
    root: Rank = 0

    def begin(self, rank: Rank) -> ScatterState:
        return ScatterState(root=self.root)

    def enter(self, state: CollectiveState, rank: Rank) -> None:
        if not isinstance(state, ScatterState):
            raise TypeError(f"expected ScatterState, got {type(state).__name__}")
        if state.root != self.root:
            raise FlockCollectiveMismatch(
                f"rank {rank} called scatter with root {self.root}, "
                f"but other ranks in the group already started with root {state.root}."
            )
        if rank == self.root:
            if self.values is None:
                raise FlockUsageError("scatter requires values on the root rank.")
            state.values = list(self.values)
        elif self.values is not None:
            raise FlockUsageError("only the scatter root rank should provide values.")


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
        return [self.values[member] for member in sorted(members)]


@dataclass
class AllReduceState:
    kind: ClassVar[str] = "all_reduce"
    op: ReduceOp
    values: dict[Rank, Any] = field(default_factory=dict)

    def complete(self, members: Group, rank: Rank) -> Any:
        ordered = [self.values[member] for member in sorted(members)]
        return reduce_values(self.op, ordered)


@dataclass
class ScatterState:
    kind: ClassVar[str] = "scatter"
    root: Rank
    values: list[Any] | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        if self.root not in members:
            raise FlockUsageError(f"scatter root {self.root} is not in the group.")
        if self.values is None:
            raise FlockUsageError("scatter root did not provide values.")
        members_sorted = sorted(members)
        if len(self.values) != len(members_sorted):
            raise FlockUsageError(
                f"scatter root provided {len(self.values)} values, "
                f"but the group has {len(members_sorted)} members."
            )
        return self.values[members_sorted.index(rank)]


def reduce_values(op: ReduceOp, values: Sequence[Any]) -> Any:
    match op:
        case "sum":
            return sum(values)
        case "prod":
            result = 1
            for value in values:
                result *= value
            return result
        case "min":
            return min(values)
        case "max":
            return max(values)
        case _:
            raise TypeError(f"unknown reduce op: {op!r}")
