from collections.abc import Sequence

Rank = int

Group = frozenset[Rank]

WORLD: Group | None = None


def new_group(ranks: Sequence[Rank]) -> Group:
    return frozenset(ranks)
