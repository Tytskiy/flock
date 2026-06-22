import torch

import flock
import flock.torch as ft


def test_tensor_send_recv_clones_payload():
    @flock.distribute(workers=2)
    async def run():
        if flock.get_rank() == 0:
            tensor = torch.tensor([1, 2])
            await ft.send(1, tensor)
            tensor.add_(10)
            return None
        return await ft.recv(0)

    result = run()
    assert result[0] is None
    assert torch.equal(result[1], torch.tensor([1, 2]))


def test_tensor_all_reduce_sum():
    @flock.distribute(workers=4)
    async def run():
        tensor = torch.tensor([flock.get_rank() + 1])
        return await ft.all_reduce(tensor).wait()

    assert all(torch.equal(value, torch.tensor([10])) for value in run())


def test_tensor_all_reduce_custom_op():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        tensor = torch.tensor([rank, 3 - rank])
        return await ft.all_reduce(tensor, torch.minimum).wait()

    assert all(torch.equal(value, torch.tensor([0, 0])) for value in run())


def test_tensor_reduce_sum_to_dst():
    @flock.distribute(workers=4)
    async def run():
        tensor = torch.tensor([flock.get_rank() + 1])
        return await ft.reduce(tensor, dst=2).wait()

    result = run()
    assert result[0] is None
    assert result[1] is None
    assert torch.equal(result[2], torch.tensor([10]))
    assert result[3] is None


def test_tensor_reduce_scatter_sum():
    @flock.distribute(workers=4)
    async def run():
        rank = flock.get_rank()
        world = flock.get_world_size()
        tensors = [torch.tensor([rank + i]) for i in range(world)]
        return await ft.reduce_scatter(tensors).wait()

    expected = [torch.tensor([6]), torch.tensor([10]), torch.tensor([14]), torch.tensor([18])]
    assert all(torch.equal(got, want) for got, want in zip(run(), expected, strict=True))
