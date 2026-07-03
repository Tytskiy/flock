from collections.abc import Callable, Sequence
from typing import Any

from flock.errors import FlockUsageError

_registry: dict[type, Callable[[Any], int]] = {}


def register_payload_bytes(typ: type | tuple[type, ...], fn: Callable[[Any], int]) -> None:
    types = (typ,) if isinstance(typ, type) else typ
    for registered in types:
        if registered in _registry:
            continue
        _registry[registered] = fn


def payload_bytes(value: Any) -> int:
    if value is None:
        return 0
    for typ in type(value).__mro__:
        fn = _registry.get(typ)
        if fn is not None:
            return fn(value)
    raise FlockUsageError(
        f"no payload bytes function registered for {type(value).__name__}.\n"
        "Register one with flock.register_payload_bytes(...), or import flock.torch for tensors."
    )


def _sequence_bytes(value: Sequence[Any]) -> int:
    return sum(payload_bytes(item) for item in value)


register_payload_bytes(str, lambda value: len(value.encode("utf-8")))
register_payload_bytes(bool, lambda _: 1)
register_payload_bytes(int, lambda _: 8)
register_payload_bytes(float, lambda _: 8)
register_payload_bytes(bytes, len)
register_payload_bytes(bytearray, len)
register_payload_bytes(memoryview, len)
register_payload_bytes(list, _sequence_bytes)
register_payload_bytes(tuple, _sequence_bytes)
