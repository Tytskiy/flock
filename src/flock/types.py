from collections.abc import Sequence
from dataclasses import dataclass

Rank = int

RequestId = int


@dataclass(frozen=True)
class Group:
    ranks: tuple[int, ...]
    id: int

    def __iter__(self):
        return iter(self.ranks)

    def __len__(self):
        return len(self.ranks)

    def index(self, idx):
        return self.ranks.index(idx)


class _World(Group):
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = Group((), -1)

        return cls.instance


WORLD: Group = _World()


class _GroupId(int):
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = 0

            return cls.instance

        cls.instance += 1
        return cls.instance


def new_group(ranks: Sequence[Rank]) -> Group:
    assert len(set(ranks)) == len(ranks)
    ranks = tuple(sorted(ranks))

    id = _GroupId()

    return Group(ranks=ranks, id=id)


def _default_world_group(world_size: int) -> Group:
    return Group(tuple(i for i in range(world_size)), 0)
