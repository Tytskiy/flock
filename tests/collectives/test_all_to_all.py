import flock


def test_all_to_all():
    @flock.distribute(workers=3)
    async def run():
        rank = flock.get_rank()
        values = [(rank, peer) for peer in range(3)]
        return await flock.all_to_all(values).wait()

    assert run() == [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
    ]
