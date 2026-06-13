import pytest

import flock
from flock.scheduler import Fifo, Random


@pytest.mark.parametrize("seed", range(8))
def test_results_are_deterministic_across_seeds(run_scheduler, seed):
    world = 5

    async def worker(rank):
        await flock.isend((rank + 1) % world, rank)
        return await flock.recv((rank - 1) % world)

    results = run_scheduler([worker(r) for r in range(world)], world, policy=Random(seed=seed))
    assert results == [(r - 1) % world for r in range(world)]


@pytest.mark.parametrize("policy", [Random(seed=0), Fifo()])
def test_policies_agree_on_results(run_scheduler, policy):
    world = 4

    async def worker(rank):
        await flock.isend((rank + 1) % world, rank)
        return await flock.recv((rank - 1) % world)

    results = run_scheduler([worker(r) for r in range(world)], world, policy=policy)
    assert results == [(r - 1) % world for r in range(world)]
