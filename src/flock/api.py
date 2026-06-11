from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Coroutine, Generator
from typing import Any

from flock.context import make_context
from flock.errors import FlockUsageError
from flock.ops import ISendOp, Op, RecvOp, SendOp
from flock.scheduler import Random, Scheduler, Worker


class FlockAwaitable[OpT: Op]:
    def __init__(self, op: OpT) -> None:
        self._op = op
        self._awaited = False

    def __await__(self) -> Generator[OpT, Any, Any]:
        self._awaited = True
        return (yield self._op)

    def __del__(self) -> None:
        if not self._awaited:
            warnings.warn(
                f"flock.{self._op.name}(...) was created but never awaited",
                RuntimeWarning,
                stacklevel=2,
            )


def isend[T](dst: int, value: T) -> FlockAwaitable[ISendOp[T]]:
    return FlockAwaitable(ISendOp(dst=dst, value=value))


def send[T](dst: int, value: T) -> FlockAwaitable[SendOp[T]]:
    return FlockAwaitable(SendOp(dst=dst, value=value))


def recv(src: int) -> FlockAwaitable[RecvOp]:
    return FlockAwaitable(RecvOp(src=src))


def distribute[R](
    workers: int,
    seed: int | None = 0,
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
            return Scheduler(spawned, policy=Random(seed=seed)).run()

        return wrapper

    return decorator
