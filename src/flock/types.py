from collections.abc import Sequence
from dataclasses import dataclass

Rank = int

RequestId = int


@dataclass(frozen=True)
class Group:
    ranks: tuple[int]
    id: int

    def __iter__(self):
        return iter(self.ranks)

    def __len__(self):
        return len(self.ranks)

    def index(self, idx):
        return self.ranks.index(idx)


class _World:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = Group([], -1)

        return cls.instance


WORLD: Group = _World()


def new_group(ranks: Sequence[Rank]) -> Group:
    if getattr(new_group, "group_id", None) is None:
        new_group.group_id = 1

    assert len(set(ranks)) == len(ranks)
    ranks = tuple(sorted(ranks))

    id = new_group.group_id
    new_group.group_id += 1

    return Group(ranks=ranks, id=id)


def _default_world_group(world_size: int) -> Group:
    return Group(tuple(i for i in range(world_size)), 0)
