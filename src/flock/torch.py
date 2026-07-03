from collections.abc import Sequence

try:
    import torch
except ImportError as exc:
    raise ImportError("flock.torch requires PyTorch. Install it with `pip install 'flock[torch]'`.") from exc

import flock
from flock.collectives import WORLD, Group, ReduceOp
from flock.collectives.ops import ReduceFn
from flock.payload import register_payload_bytes
from flock.types import Rank
from flock.work import Work

register_payload_bytes(torch.Tensor, lambda tensor: tensor.numel() * tensor.element_size())


def isend(dst: Rank, tensor: torch.Tensor, tag: int = 0) -> Work[None]:
    return flock.isend(dst, tensor.clone(), tag=tag)


def irecv(src: Rank, tag: int = 0) -> Work[torch.Tensor]:
    return flock.irecv(src, tag=tag, expected_type=torch.Tensor)


async def send(dst: Rank, tensor: torch.Tensor, tag: int = 0) -> None:
    await flock.send(dst, tensor.clone(), tag=tag)


async def recv(src: Rank, tag: int = 0) -> torch.Tensor:
    return await flock.recv(src, tag=tag, expected_type=torch.Tensor)


def barrier(group: Group = WORLD) -> Work[None]:
    return flock.barrier(group=group)


def all_gather(tensor: torch.Tensor, group: Group = WORLD) -> Work[list[torch.Tensor]]:
    return flock.all_gather(tensor.clone(), group=group)


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp | str | ReduceFn = ReduceOp.SUM,
    group: Group = WORLD,
) -> Work[torch.Tensor]:
    return flock.all_reduce(tensor.clone(), op=op, group=group)


def reduce(
    tensor: torch.Tensor,
    op: ReduceOp | str | ReduceFn = ReduceOp.SUM,
    dst: Rank = 0,
    group: Group = WORLD,
) -> Work[torch.Tensor | None]:
    return flock.reduce(tensor.clone(), op=op, dst=dst, group=group)


def broadcast(tensor: torch.Tensor | None, src: Rank = 0, group: Group = WORLD) -> Work[torch.Tensor]:
    value = tensor.clone() if tensor is not None else None
    return flock.broadcast(value, src=src, group=group)


def gather(tensor: torch.Tensor, dst: Rank = 0, group: Group = WORLD) -> Work[list[torch.Tensor] | None]:
    return flock.gather(tensor.clone(), dst=dst, group=group)


def scatter(
    tensors: Sequence[torch.Tensor] | None,
    src: Rank = 0,
    group: Group = WORLD,
) -> Work[torch.Tensor]:
    values = [tensor.clone() for tensor in tensors] if tensors is not None else None
    return flock.scatter(values, src=src, group=group)


def reduce_scatter(
    tensors: Sequence[torch.Tensor],
    op: ReduceOp | str | ReduceFn = ReduceOp.SUM,
    group: Group = WORLD,
) -> Work[torch.Tensor]:
    return flock.reduce_scatter([tensor.clone() for tensor in tensors], op=op, group=group)


def all_to_all(tensors: Sequence[torch.Tensor], group: Group = WORLD) -> Work[list[torch.Tensor]]:
    return flock.all_to_all([tensor.clone() for tensor in tensors], group=group)
