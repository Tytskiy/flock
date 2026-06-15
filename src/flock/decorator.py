import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import Any

from flock.context import make_context
from flock.errors import FlockUsageError
from flock.scheduler import CooperativeScheduler, Policy, Random, Worker


def distribute[R](
    workers: int,
    seed: int | None = 0,
    policy: Policy | None = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., list[R]]]:
    if callable(workers):
        raise FlockUsageError(
            "@flock.distribute needs to know how many workers to run.\n"
            "It looks like you forgot the parentheses. Write it like this:\n\n"
            "    @flock.distribute(workers=4)\n"
            "    async def run():\n"
            "        ..."
        )

    if not isinstance(workers, int) or workers < 1:
        raise FlockUsageError(f"workers must be a whole number greater than 0, but you gave {workers!r}.")

    if policy is not None and not isinstance(policy, Policy):
        raise FlockUsageError(
            f"policy must be a scheduling policy like Random() or Fifo(), but you gave {policy!r}."
        )

    chosen = policy if policy is not None else Random(seed=seed)

    def decorator(fn: Callable[..., Coroutine[Any, Any, R]]) -> Callable[..., list[R]]:
        if inspect.isasyncgenfunction(fn):
            raise FlockUsageError(
                f"@flock.distribute can't run {fn.__name__!r} because it has a `yield` "
                "inside.\n"
                "Remove the `yield` and return a value (or nothing) instead."
            )

        if not inspect.iscoroutinefunction(fn):
            raise FlockUsageError(
                f"@flock.distribute needs an `async def`, but {fn.__name__!r} is a normal "
                "function.\n"
                "Add `async` in front so it can pause and wait for messages:\n\n"
                "    @flock.distribute(workers=4)\n"
                "    async def run():\n"
                "        ..."
            )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> list[R]:
            spawned = [
                Worker(
                    coro=fn(*args, **kwargs),
                    context=make_context(rank, workers),
                )
                for rank in range(workers)
            ]
            try:
                return CooperativeScheduler(spawned, policy=chosen).run()
            except BaseException:
                for worker in spawned:
                    worker.coro.close()
                raise

        return wrapper

    return decorator
