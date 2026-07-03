import pytest
import torch

import flock
import flock.torch as ft
from flock import FlockUsageError
from flock.payload import payload_bytes, register_payload_bytes


def test_builtin_payload_sizers():
    assert payload_bytes(None) == 0
    assert payload_bytes("hi") == 2
    assert payload_bytes(b"abc") == 3
    assert payload_bytes(True) == 1
    assert payload_bytes(42) == 8
    assert payload_bytes(3.14) == 8


def test_unregistered_type_raises():
    with pytest.raises(FlockUsageError, match="no payload bytes function registered"):
        payload_bytes(object())


def test_torch_tensor_registered_on_flock_torch_import():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert payload_bytes(tensor) == tensor.numel() * tensor.element_size()


def test_custom_payload_sizer():
    class Blob:
        def __init__(self, size: int) -> None:
            self.size = size

    register_payload_bytes(Blob, lambda blob: blob.size)
    assert payload_bytes(Blob(17)) == 17


def test_duplicate_registration_is_noop():
    class Widget:
        pass

    register_payload_bytes(Widget, lambda _: 1)
    register_payload_bytes(Widget, lambda _: 999)
    assert payload_bytes(Widget()) == 1


def test_flock_torch_reload_does_not_raise():
    import importlib

    importlib.reload(ft)
    tensor = torch.tensor([1.0, 2.0])
    assert payload_bytes(tensor) == tensor.numel() * tensor.element_size()


def test_tracer_uses_tensor_bytes():
    @flock.distribute(workers=2, trace=True)
    async def ping():
        if flock.get_rank() == 0:
            await ft.send(1, torch.tensor([1.0, 2.0]))
            return None
        return await ft.recv(0)

    _, trace = ping()
    assert trace.summary().total_bytes == 2 * torch.tensor([1.0]).element_size()
