import torch.distributed as dist



def get_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank

def print_rank_0(message):
    rank = get_rank()
    if rank == 0:
        print(f"0 - {message}", flush=True)

def print_rank_all(message):
    rank = get_rank()
    print(f"{rank} - {message}", flush=True)

def stage_log(stage):
    rank = get_rank()
    message = f"==========={rank} - {stage}==========="
    print(message, flush=True)