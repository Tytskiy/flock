from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from flock.context import get_rank as get_current_rank
from flock.errors import FlockUsageError
from flock.p2p.handle import P2PHandle
from flock.p2p.ops import Irecv, Isend, P2PCall, Recv, Send
from flock.payload import payload_bytes
from flock.scheduler.port import SchedulePort
from flock.tracer import Tracer
from flock.types import ANY_SOURCE, ANY_TAG, Rank, RequestId


@dataclass
class Message:
    src: Rank
    value: Any
    tag: int = 0
    ack: bool = False
    send_id: int | None = None


@dataclass
class P2PRequest:
    handle: P2PHandle
    done: bool = False
    message: Message | None = None


class P2PEngine:
    def __init__(self, port: SchedulePort, *, tracer: Tracer | None = None) -> None:
        self._port = port
        self._tracer = tracer
        self.mailboxes: defaultdict[Rank, deque[Message]] = defaultdict(deque)
        self.requests: dict[RequestId, P2PRequest] = {}
        self.blocked: dict[Rank, P2PHandle] = {}
        self._request_counter = 0

    def begin(self, call: P2PCall) -> P2PHandle:
        rank = get_current_rank()
        self._validate_peer(call)
        handle = self._new_handle(call.kind, rank, call.peer, call.tag)

        match call:
            case Isend(value=value):
                if self._tracer is not None:
                    self._tracer.p2p_begin(
                        rank,
                        call.kind,
                        call.peer,
                        call.tag,
                        nbytes=payload_bytes(value),
                    )
                self._deliver(rank, call.peer, value, call.tag, ack=False)
                self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)

            case Send(value=value):
                if self._tracer is not None:
                    self._tracer.p2p_begin(
                        rank,
                        call.kind,
                        call.peer,
                        call.tag,
                        nbytes=payload_bytes(value),
                    )
                self.requests[handle.request_id] = P2PRequest(handle=handle)
                self._deliver(rank, call.peer, value, call.tag, ack=True, send_id=handle.request_id)

            case Recv() | Irecv():
                request = P2PRequest(handle=handle)
                self.requests[handle.request_id] = request
                message = self._take_message(rank, handle)
                if message is not None:
                    self._bind_receive(request, message)

            case _:
                raise TypeError(f"unknown p2p call: {call!r}")

        return handle

    def wait(self, rank: Rank, handle: P2PHandle) -> None:
        request = self.requests.get(handle.request_id)
        if request is None:
            raise FlockUsageError(
                f"rank {rank} tried to wait on {handle.kind} without starting it on this rank."
            )

        match handle.kind:
            case "isend":
                self.requests.pop(handle.request_id)
                self._port.resume(rank)

            case "send":
                if request.done:
                    self.requests.pop(handle.request_id)
                    self._port.resume(rank)
                else:
                    self.blocked[rank] = handle

            case "recv" | "irecv":
                if not request.done:
                    self.blocked[rank] = handle
                    return

                self.requests.pop(handle.request_id)
                assert request.message is not None
                message = request.message
                self._port.resume(rank, message.value)

    def deadlock_lines(self) -> list[str]:
        lines = []
        for rank, handle in sorted(self.blocked.items()):
            peer = "any source" if handle.peer == ANY_SOURCE else f"rank {handle.peer}"
            lines.append(f"rank {rank} is blocked in {handle.kind} waiting for {peer}")
        return lines

    def is_complete(self, handle: P2PHandle) -> bool:
        request = self.requests.get(handle.request_id)
        if request is None:
            return True
        return request.done

    def _new_handle(self, kind: str, rank: Rank, peer: Rank, tag: int) -> P2PHandle:
        request_id = self._request_counter
        self._request_counter += 1
        return P2PHandle(kind=kind, rank=rank, peer=peer, tag=tag, request_id=request_id)

    def _validate_peer(self, call: P2PCall) -> None:
        peer = call.peer
        if call.kind in ("recv", "irecv") and peer == ANY_SOURCE:
            return
        world_size = self._port.world_size
        if peer < 0 or peer >= world_size:
            raise FlockUsageError(f"rank {peer} is out of range for world_size={world_size}.")

    def _deliver(
        self, src: Rank, dst: Rank, value: Any, tag: int, *, ack: bool, send_id: int | None = None
    ) -> None:
        if self._tracer is not None:
            self._tracer.p2p_deliver(src, dst, nbytes=payload_bytes(value), tag=tag)

        receiver = self._posted_recv(dst, src, tag)
        if receiver is not None:
            self._bind_receive(
                receiver,
                Message(src=src, value=value, tag=tag, ack=ack, send_id=send_id),
            )
            return

        self.mailboxes[dst].append(Message(src=src, value=value, tag=tag, ack=ack, send_id=send_id))

    def _posted_recv(self, dst: Rank, src: Rank, tag: int) -> P2PRequest | None:
        incoming = Message(src=src, value=None, tag=tag)
        # Dict insertion order is receive posting order.
        for request in self.requests.values():
            handle = request.handle
            if (
                not request.done
                and handle.rank == dst
                and handle.kind in ("recv", "irecv")
                and self._matches(handle, incoming)
            ):
                return request
        return None

    def _bind_receive(self, request: P2PRequest, message: Message) -> None:
        request.done = True
        request.message = message
        if message.ack and message.send_id is not None:
            self._complete_send(message.send_id)

        handle = request.handle
        blocked = self.blocked.get(handle.rank)
        if blocked is None or blocked.request_id != handle.request_id:
            return
        del self.blocked[handle.rank]
        self.requests.pop(handle.request_id)
        self._port.resume(handle.rank, message.value)

    def _take_message(self, dst: Rank, handle: P2PHandle) -> Message | None:
        mailbox = self.mailboxes.get(dst)
        if not mailbox:
            return None
        for index, message in enumerate(mailbox):
            if self._matches(handle, message):
                del mailbox[index]
                return message
        return None

    @staticmethod
    def _matches(handle: P2PHandle, message: Message) -> bool:
        src_ok = handle.peer == ANY_SOURCE or handle.peer == message.src
        tag_ok = handle.tag == ANY_TAG or handle.tag == message.tag
        return src_ok and tag_ok

    def _complete_send(self, send_id: int) -> None:
        request = self.requests.get(send_id)
        if request is None:
            return

        request.done = True
        rank = request.handle.rank
        if rank in self.blocked and self.blocked[rank].request_id == send_id:
            del self.blocked[rank]
            self.requests.pop(send_id)
            self._port.resume(rank)
