import flock


@flock.distribute(workers=2)
async def stuck():
    rank = flock.get_rank()
    if rank == 0:
        return await flock.recv(1)

    return "rank 1 never sends"


stuck()
