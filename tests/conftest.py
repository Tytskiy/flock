from collections.abc import Coroutine
from typing import Any

import pytest

from flock.context import make_context
from flock.scheduler import CooperativeScheduler, Policy, Random, Worker


@pytest.fixture
def run_scheduler():
    def _run(
        coros: list[Coroutine[Any, Any, object]],
        world_size: int,
        *,
        policy: Policy | None = None,
    ) -> list[object]:
        if policy is None:
            policy = Random(seed=0)
        workers = [Worker(coro, context=make_context(rank, world_size)) for rank, coro in enumerate(coros)]
        try:
            return CooperativeScheduler(workers, policy=policy).run()
        except BaseException:
            for worker in workers:
                worker.coro.close()
            raise

    return _run
