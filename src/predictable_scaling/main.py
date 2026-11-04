import os
import torch
import torch.distributed as dist

from transformers import LlamaConfig, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel

from predictable_scaling.optimizer import muon
from predictable_scaling.dataloader import load_dataset
from predictable_scaling.model import build_model
from predictable_scaling.train import train

class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

def setup_distributed() -> DistInfo:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return DistInfo(
        rank = dist.get_rank(),
        world_size = dist.get_world_size(),
        local_rank = local_rank,
        device = torch.device("cuda", local_rank)
    )

def main():
    info = setup_distributed()

    try:
        model = build_model().to(info.device)
        opt = muon(model)
        loader = load_dataset()
        model = DistributedDataParallel(model, device_ids=[info.local_rank])
        train(model, opt, loader, info)

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()