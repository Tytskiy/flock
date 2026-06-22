from collections.abc import Sequence

from flock.collectives.ops import NewGroup
from flock.context import get_world_size
from flock.types import WORLD, Group, Rank


async def new_group(ranks: Sequence[Rank]) -> Group:
    from flock.scheduler.runtime import require_runtime
    from flock.work import Work

    runtime = require_runtime()
    return await Work(
        runtime.begin_collective(WORLD, NewGroup(ranks, world_size=get_world_size())),
        runtime,
        expected_type=Group,
    ).wait()
