import flock


@flock.distribute(workers=4)
async def even_group():
    rank = flock.get_rank()
    even_ranks = await flock.new_group([0, 2])

    if rank not in even_ranks:
        return None

    return await flock.all_gather(f"rank {rank}", group=even_ranks).wait()


print(even_group())
