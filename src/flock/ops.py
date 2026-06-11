from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


class Op:
    name: ClassVar[str]


@dataclass
class ISendOp[T](Op):
    name: ClassVar[str] = "isend"
    dst: int
    value: T


@dataclass
class SendOp[T](Op):
    name: ClassVar[str] = "send"
    dst: int
    value: T


@dataclass
class RecvOp(Op):
    name: ClassVar[str] = "recv"
    src: int
