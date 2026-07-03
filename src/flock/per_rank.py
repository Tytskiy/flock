from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PerRank[T]:
    values: Sequence[T]


def per_rank[T](values: Sequence[T]) -> PerRank[T]:
    return PerRank(values)
