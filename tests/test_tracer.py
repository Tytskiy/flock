import flock
from flock import P2PBegin, P2PDeliver, payload_bytes


def test_distribute_trace_returns_tuple():
    @flock.distribute(workers=2, trace=True)
    async def run():
        return flock.get_rank()

    result = run()
    assert isinstance(result, tuple)
    results, trace = result
    assert results == [0, 1]
    assert isinstance(trace, flock.Tracer)


def test_distribute_without_trace_returns_list():
    @flock.distribute(workers=2)
    async def run():
        return flock.get_rank()

    assert run() == [0, 1]


def test_tracer_counts_ping_bytes():
    @flock.distribute(workers=2, trace=True)
    async def ping():
        rank = flock.get_rank()
        if rank == 0:
            await flock.send(1, "hi")
            return None
        await flock.recv(0)
        return None

    _, trace = ping()
    summary = trace.summary()
    assert summary.total_messages == 1
    assert summary.total_bytes == 2
    assert summary.bytes_sent == (2, 0)
    assert summary.bytes_received == (0, 2)


def test_tracer_ring_shift_bytes():
    @flock.distribute(workers=4, trace=True)
    async def ring_shift():
        rank = flock.get_rank()
        world = flock.get_world_size()
        dst = (rank + 1) % world
        src = (rank - 1) % world
        await flock.isend(dst, rank).wait()
        return await flock.recv(src)

    _, trace = ring_shift()
    summary = trace.summary()
    assert summary.total_messages == 4
    assert summary.total_bytes == 4 * payload_bytes(0)
    assert summary.bytes_sent == (8, 8, 8, 8)


def test_tracer_collective_sync():
    @flock.distribute(workers=2, trace=True)
    async def gather_builtin():
        work = flock.all_gather(flock.get_rank())
        return await work.wait()

    _, trace = gather_builtin()
    summary = trace.summary()
    assert summary.total_messages == 0
    assert summary.total_bytes == 0
    assert summary.collective_counts == (("all_gather", 1),)


def test_tracer_events_are_typed():
    @flock.distribute(workers=2, trace=True)
    async def ping():
        if flock.get_rank() == 0:
            await flock.send(1, "x")
        else:
            await flock.recv(0)

    _, trace = ping()
    kinds = {type(event) for event in trace.events}
    assert P2PBegin in kinds
    assert P2PDeliver in kinds


def test_tracer_str_contains_totals():
    @flock.distribute(workers=2, trace=True)
    async def ping():
        if flock.get_rank() == 0:
            await flock.send(1, "x")
        else:
            await flock.recv(0)

    _, trace = ping()
    text = str(trace)
    assert "communication trace" in text
    assert "messages  1" in text
    assert "collectives" in text
    assert "none" in text


def test_tracer_report_matrix_uses_dot_for_zero():
    @flock.distribute(workers=2, trace=True)
    async def ping():
        if flock.get_rank() == 0:
            await flock.send(1, "x")
        else:
            await flock.recv(0)

    _, trace = ping()
    text = str(trace)
    assert "0 →" in text
    assert "·" in text


def test_tracer_timeline():
    @flock.distribute(workers=2, trace=True, policy=flock.Fifo())
    async def ping():
        if flock.get_rank() == 0:
            await flock.send(1, "x")
        else:
            await flock.recv(0)

    _, trace = ping()
    text = trace.timeline()
    assert "timeline" in text
    assert "begin" in text
    assert "rank 0 send→1" in text
    assert "deliver" in text
    assert "0→1" in text
    assert "schedule" not in text
    assert "block" not in text
    assert "resume" not in text
