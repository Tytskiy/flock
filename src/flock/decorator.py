import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, Literal, overload

from flock.context import make_context
from flock.errors import FlockUsageError
from flock.per_rank import PerRank
from flock.scheduler import CooperativeScheduler, Policy, Random, Worker
from flock.tracer import Tracer


def _validate_per_rank(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    workers: int,
    fn_name: str,
) -> None:
    for index, arg in enumerate(args):
        if isinstance(arg, PerRank) and len(arg.values) != workers:
            raise FlockUsageError(
                f"{fn_name} got per_rank(...) with {len(arg.values)} values at argument "
                f"position {index}, but @flock.distribute(workers={workers}) runs "
                f"{workers} workers.\n"
                "Pass one value per worker."
            )

    for name, value in kwargs.items():
        if isinstance(value, PerRank) and len(value.values) != workers:
            raise FlockUsageError(
                f"{fn_name} got per_rank(...) with {len(value.values)} values for "
                f"argument {name!r}, but @flock.distribute(workers={workers}) runs "
                f"{workers} workers.\n"
                "Pass one value per worker."
            )


def _localize(value: Any, rank: int) -> Any:
    if isinstance(value, PerRank):
        return value.values[rank]
    return value


def _localize_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rank: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (
        tuple(_localize(arg, rank) for arg in args),
        {name: _localize(value, rank) for name, value in kwargs.items()},
    )


@overload
def distribute[R](
    workers: int,
    seed: int | None = 0,
    policy: Policy | None = None,
    *,
    trace: Literal[True],
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., tuple[list[R], Tracer]]]: ...


@overload
def distribute[R](
    workers: int,
    seed: int | None = 0,
    policy: Policy | None = None,
    *,
    trace: Literal[False] = False,
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., list[R]]]: ...


def distribute[R](
    workers: int,
    seed: int | None = 0,
    policy: Policy | None = None,
    *,
    trace: bool = False,
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., list[R] | tuple[list[R], Tracer]]]:
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

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, R]],
    ) -> Callable[..., list[R] | tuple[list[R], Tracer]]:
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
        def wrapper(*args: Any, **kwargs: Any) -> list[R] | tuple[list[R], Tracer]:
            _validate_per_rank(args, kwargs, workers=workers, fn_name=fn.__name__)
            spawned = []
            for rank in range(workers):
                local_args, local_kwargs = _localize_call(
                    args,
                    kwargs,
                    rank=rank,
                )
                spawned.append(
                    Worker(
                        coro=fn(*local_args, **local_kwargs),
                        context=make_context(rank, workers),
                    )
                )
            tracer = Tracer(world_size=workers) if trace else None
            try:
                results = CooperativeScheduler(spawned, policy=chosen, tracer=tracer).run()
            except BaseException:
                for worker in spawned:
                    worker.coro.close()
                raise

            if tracer is not None:
                return results, tracer
            return results

        return wrapper

    return decorator
