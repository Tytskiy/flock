import flock


@flock.distribute(workers=4)
async def ring():
    rank = flock.get_rank()
    world = flock.get_world_size()

    await flock.isend((rank + 1) % world, f"hello from {rank}").wait()
    return await flock.recv((rank - 1) % world)


print(ring())
