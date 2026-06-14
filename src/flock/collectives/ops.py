from __future__ import annotations

import copy
from collections.abc import Sequence
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

        state.value = reduce_value(self.op, state.value, self.value)


@dataclass(frozen=True)
class Scatter:
    kind: ClassVar[str] = "scatter"
    values: Sequence[Any] | None
    src: Rank = 0

    def begin(self, rank: Rank) -> ScatterState:
        return ScatterState(src=self.src)

    def enter(self, state: CollectiveState, rank: Rank) -> None:
        if not isinstance(state, ScatterState):
            raise TypeError(f"expected ScatterState, got {type(state).__name__}")
        if state.src != self.src:
            raise FlockCollectiveMismatch(
                f"rank {rank} called scatter with src {self.src}, "
                f"but other ranks in the group already started with src {state.src}."
            )
        if rank == self.src:
            if self.values is None:
                raise FlockUsageError("scatter requires values on the src rank.")
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
    op: ReduceOp
    value: Any | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        return self.value


@dataclass
class ScatterState:
    kind: ClassVar[str] = "scatter"
    src: Rank
    values: list[Any] | None = None

    def complete(self, members: Group, rank: Rank) -> Any:
        if self.src not in members:
            raise FlockUsageError(f"scatter src {self.src} is not in the group.")
        if self.values is None:
            raise FlockUsageError("scatter src did not provide values.")
        if len(self.values) != len(members):
            raise FlockUsageError(
                f"scatter src provided {len(self.values)} values, but the group has {len(members)} members."
            )
        return self.values[members.index(rank)]


def reduce_value(op: ReduceOp, acc: Any | None, curr: Any) -> Any:
    if acc is None:
        return curr
    match op:
        case ReduceOp.SUM:
            return acc + curr
        case ReduceOp.PROD:
            return acc * curr
        case ReduceOp.MIN:
            return min(acc, curr)
        case ReduceOp.MAX:
            return max(acc, curr)
        case _:
            raise TypeError(f"unknown reduce op: {op!r}")
