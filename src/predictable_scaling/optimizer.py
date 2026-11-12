from transformers import AutoModelForCausalLM
from muonium import Muon
import torch.distributed as dist
import torch

class DistributedMuon(Muon):
    """Muon where each rank orthogonalizes only the matrices it owns, then
    broadcasts the updated weights. Same result as plain Muon under DDP
    (grads are already averaged), with ~1/world_size of the optimizer compute."""

    def __init__(self, muon_params, adam_params, *, muon_lr, adam_lr, adam_betas, wd, **kwargs):
        dist_on = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_on else 0
        world = dist.get_world_size() if dist_on else 1

        # sort by how big a parameter is and round robbin to ranks. load balance
        order = sorted(range(len(muon_params)), key = lambda i: -muon_params[i].numel())
        owner = [0] * len(muon_params) # record who which rank own a parameter matrix

        for k, i in enumerate(order):
            owner[i] = k % world
        mine = [p for p, o in zip(muon_params, owner) if o == rank]

        super().__init__(
            [
                {"params": mine, "lr": muon_lr},
                {"params": adam_params, "algorithm": "adamw", "lr": adam_lr, "betas": adam_betas},
            ],
            wd=wd,
            **kwargs
        )

        self._dist_on = dist_on and world > 1
        self._muon_owner = list(zip(muon_params, owner))

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._dist_on:
            works = [dist.broadcast(p.data, src=src, async_op=True) for p, src in self._muon_owner]
            for w in works:
                w.wait()
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)

        for p, _ in self._muon_owner:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach().zero_()

def muon(model: AutoModelForCausalLM, adam_lr=1e-3, adam_beta1=0.9,adam_beta2=0.95, muon_lr=1e-2, muon_momentum=0.95, wd=0.1, adam_eps=1e-8) -> Muon:
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    optimizer = DistributedMuon(muon_params, adam_params, muon_lr=muon_lr, momentum=muon_momentum, adam_lr=adam_lr, adam_betas=(adam_beta1, adam_beta2), wd=wd, adam_eps=adam_eps)

    return optimizer