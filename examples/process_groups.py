import flock

EVEN_RANKS = flock.new_group([0, 2])


@flock.distribute(workers=4)
async def even_group():
    rank = flock.get_rank()

    if rank not in EVEN_RANKS:
        return None

    return await flock.all_gather(f"rank {rank}", group=EVEN_RANKS).wait()


print(even_group())
