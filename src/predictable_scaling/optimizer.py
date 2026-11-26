import torch

from transformers import AutoModelForCausalLM
from muonium import Muon
from torch import Tensor
from torch.optim import Optimizer
import torch.distributed as dist

LINEAR_STATE_KEYS = {"exp_avg", "momentum_buffer", "grad_avg"}

class DistributedMuon(Muon):
    """Muon where each rank orthogonalizes only the matrices it owns, then
    broadcasts the updated weights. Same result as plain Muon under DDP
    (grads are already averaged), with ~1/world_size of the optimizer compute."""

    def __init__(self, groups: list[dict], all_muon_params: list, **kwargs):
        """groups: muonium param-group dicts, each with a "name" and an "algorithm".
        all_muon_params: every muon parameter in model order. Ownership is decided over
        this list (not per group), so regrouping never changes which rank owns a matrix
        and old per-rank optimizer checkpoints stay valid."""
        dist_on = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_on else 0
        world = dist.get_world_size() if dist_on else 1

        # sort by how big a parameter is and round robbin to ranks. load balance
        order = sorted(range(len(all_muon_params)), key = lambda i: -all_muon_params[i].numel())
        owner = [0] * len(all_muon_params) # record which rank owns a parameter matrix
        for k, i in enumerate(order):
            owner[i] = k % world
        owner_of = {id(p): o for p, o in zip(all_muon_params, owner)}

        mine = []
        for g in groups:
            g = dict(g)
            if g.get("algorithm", "muon") == "muon":
                g["params"] = [p for p in g["params"] if owner_of[id(p)] == rank]
            mine.append(g)
        name_of = {id(p): g["name"] for g in mine for p in g["params"]}

        # muonium drops empty groups (a rank may own none of a small group) and unknown keys
        super().__init__(mine, **kwargs)
        for g in self.param_groups:
            g["name"] = name_of[id(g["params"][0])]

        self._dist_on = dist_on and world > 1
        self._muon_owner = list(zip(all_muon_params, owner))

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

    groups = [
        {"name": "muon", "params": muon_params, "algorithm": "muon",
         "lr": muon_lr, "momentum": muon_momentum, "wd": wd},
        {"name": "adam", "params": adam_params, "algorithm": "adamw",
         "lr": adam_lr, "betas": (adam_beta1, adam_beta2), "eps": adam_eps, "wd": wd},
    ]
    return DistributedMuon(groups, muon_params)

def muon_from_specs(model, specs: list[dict]) -> DistributedMuon:
    """Build DistributedMuon from hparams.resolve_groups() output. model is the DDP wrapper."""
    named = dict(model.module.named_parameters())
    groups = []
    for s in specs:
        g = {k: v for k, v in s.items() if k != "param_names"}
        g["params"] = [named[n] for n in s["param_names"]]
        groups.append(g)
    muon_names = {n for s in specs if s["algorithm"] == "muon" for n in s["param_names"]}
    all_muon = [p for n, p in model.module.named_parameters() if n in muon_names]  # model order
    return DistributedMuon(groups, all_muon)

def soap(model: AutoModelForCausalLM, lr=3e-3, beta1=0.95, beta2=0.95, shampoo_beta=0.95, weight_decay=0.1, precondition_frequency=10, max_precond_dim=10000) -> SOAP:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    inner = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    optimizer = SOAP(
        inner=inner,
        shampoo_beta=shampoo_beta,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim
    )

    return optimizer

def soap_muon(model: AutoModelForCausalLM, shampoo_beta=0.95, adam_lr=1e-3, adam_beta1=0.9, adam_beta2=0.95, muon_lr=1e-2, muon_momentum=0.95, wd=0.1):
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    inner = Muon([
            {"params": muon_params, "lr": muon_lr},
            {"params": adam_params, "algorithm": "adamw", "lr": adam_lr, "betas": (adam_beta1, adam_beta2)},
            ],
            wd=wd,
            momentum=muon_momentum
        )
    for g, name in zip(inner.param_groups, ("muon", "adam")):  # for lr logging by name
        g["name"] = name
    optimizer = SOAP(inner=inner, shampoo_beta=shampoo_beta)

    return optimizer

def soap_muon_from_specs(model, specs: list[dict], soap_hp: dict) -> "SOAP":
    """SOAP around an (undistributed) Muon built from hparams.resolve_groups() output.
    model is the DDP wrapper."""
    named = dict(model.module.named_parameters())
    groups = []
    for s in specs:
        g = {k: v for k, v in s.items() if k not in ("name", "param_names")}
        g["params"] = [named[n] for n in s["param_names"]]
        groups.append(g)
    inner = Muon(groups)
    for g, s in zip(inner.param_groups, specs): # for lr logging by name
        g["name"] = s["name"]
    return SOAP(inner=inner, **soap_hp)

class Basis:
    def __init__(self, QL: Tensor, QR: Tensor):
        self.QL: Tensor = QL
        self.QR: Tensor = QR
    
    def prj(self, x: Tensor) -> Tensor:
        if self.QL is not None:
            x = self.QL.T @ x
        if self.QR is not None:
            x = x @ self.QR
        return x

    def back(self, x: Tensor) -> Tensor:
        if self.QL is not None:
            x = self.QL @ x
        if self.QR is not None:
            x = x @ self.QR.T
        return x

def eigh_basis(M: Tensor) -> Tensor:
    eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    _, Q = torch.linalg.eigh(M + 1e-30 * eye)
    return torch.flip(Q, dims=[1])

class SOAP(Optimizer):
    def __init__(
            self,
            inner: Optimizer,
            shampoo_beta: float = 0.95,
            precondition_frequency: int = 10,
            max_precond_dim: int = 10000,
            precondition = None,
            state_kinds = None
    ):
        self.inner = inner
        self.shampoo_beta = shampoo_beta
        self.precondition_frequency = precondition_frequency
        self.max_precond_dim = max_precond_dim
        self._precond = precondition or (lambda p: p.dim() >= 2)
        self._state_kinds = dict(state_kinds or {})
        self._warned = set()

        self._ready = False
        super().__init__(inner.param_groups, defaults={})
        self.param_groups = inner.param_groups
        self._ready = True
    
    def add_param_group(self, param_group):
        if self._ready:
            self.inner.add_param_group(param_group)
        else:
            super().add_param_group(param_group)

    def _init_basis(self, state: dict, G: Tensor):
        m, n = G.shape
        state["step"] = 0
        state["L"] = None
        state["R"] = None
        state["QL"] = None
        state["QR"] = None

        if m <= self.max_precond_dim:
            state["L"] = (1 - self.shampoo_beta) * (G @ G.T)
            state["QL"] = eigh_basis(state["L"])
        if n <= self.max_precond_dim:
            state["R"] = (1 - self.shampoo_beta) * (G.T @ G)
            state["QR"] = eigh_basis(state["R"])
    
    def _kind(self, key):
        if key in self._state_kinds:
            return self._state_kinds[key]
        return "linear" if key in LINEAR_STATE_KEYS else "permute"

    def _refresh_basis(self, p: Tensor, state: dict):
        old = Basis(state["QL"], state["QR"])
        orders = {}
        for st, qkey, dim in (("L", "QL", 0), ("R", "QR", 1)):
            M: Tensor = state[st]
            if M is None:
                continue
            Q: Tensor = state[qkey]
            MQ = M @ Q
            est_eig = (Q * MQ).sum(0)
            order = torch.argsort(est_eig, descending=True)
            state[qkey], _ = torch.linalg.qr(MQ[:, order])
            orders[dim] = order
        new = Basis(state["QL"], state["QR"])

        m = p.shape[0]
        for key, v in self.inner.state[p].items():
            if not torch.is_tensor(v) or v.dim() == 0:
                continue
            if v.shape != p.shape:
                if key not in self._warned:
                    self._warned.add(key)
                    continue
            
            kind = self._kind(key)
            if kind == "keep":
                continue
            if kind == "reset":
                v.zero_()
                continue
            v2 = v.reshape(m, -1).float()
            if kind == "linear":
                v2 = new.prj(old.back(v2))
            elif kind == "permute":
                for dim, order in orders.items():
                    v2 = v2.index_select(dim, order)
            else:
                raise ValueError(f"unknown state kind {kind!r} for '{key}'")
            
            v.copy_(v2.reshape(v.shape))

    @torch.no_grad()
    def step(self):
        loss = None

        shampoo_beta = self.shampoo_beta
        pending, hidden = [], []
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self._precond(p):
                    continue
                st = self.state[p]
                G = p.grad.detach().float().reshape(p.shape[0], -1)

                if len(st) == 0:
                    self._init_basis(st, G)
                    hidden.append((p, p.grad))
                    p.grad = None
                    continue

                if st["QL"] is None and st["QR"] is None:
                    continue

                B = Basis(st["QL"], st["QR"])
                W = p.detach().float().reshape(G.shape).clone()
                rot = B.prj(W)
                p.copy_(rot.reshape(p.shape))
                before = rot.to(p.dtype).float()

                orig_grad = p.grad
                p.grad = B.prj(G).reshape(p.shape).to(orig_grad.dtype)
                pending.append((p, W, before, G, orig_grad))

        try:
            self.inner.step()
        finally:
            for p, W, before, G, orig_grad in pending:
                st = self.state[p]
                B = Basis(st["QL"], st["QR"])
                delta = p.detach().float().reshape(W.shape) - before
                p.copy_((W + B.back(delta)).reshape(p.shape))
                p.grad = orig_grad
            for p, g in hidden:
                p.grad = g
        
        for p, _, _, G, _ in pending:
            st = self.state[p]
            st["step"] += 1
            if st["L"] is not None:
                st["L"].mul_(shampoo_beta).add_(G @ G.T, alpha=(1 - shampoo_beta))
            if st["R"] is not None:
                st["R"].mul_(shampoo_beta).add_(G.T @ G, alpha=(1 - shampoo_beta))
            if st["step"] % self.precondition_frequency == 0:
                self._refresh_basis(p, st)

        return loss

    