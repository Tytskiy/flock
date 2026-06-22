import flock


@flock.distribute(workers=4)
async def sum_ranks():
    rank = flock.get_rank()
    total = await flock.all_reduce(rank, "sum").wait()
    gathered = await flock.all_gather(f"rank {rank}").wait()
    return total, gathered


print(sum_ranks())
