import torch

from transformers import AutoModelForCausalLM
from muonium import Muon
from torch import Tensor
from torch.optim import Optimizer

def muon(model: AutoModelForCausalLM, adam_lr=1e-3, adam_beta1=0.9,adam_beta2=0.95, muon_lr=1e-2, wd=0.1) -> Muon:
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    optimizer = Muon(
        [
            {"params": muon_params, "lr": muon_lr},
            {"params": adam_params, "algorithm": "adamw", "lr": adam_lr, "betas": (adam_beta1, adam_beta2)},
        ],
        wd=wd
    )

    return optimizer

def soap(model: AutoModelForCausalLM, lr=3e-3, beta1=0.95, beta2=0.95, shampoo_beta=0.95, weight_decay=0.1, precondition_frequency=10, max_precond_dim=10000) -> SOAP:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    optimizer = SOAP(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ],
        lr=lr,
        betas=(beta1, beta2),
        shampoo_beta=shampoo_beta,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        weight_decay=weight_decay
    )

    return optimizer
    

class SOAP(Optimizer):
    def __init__(
            self,
            params,
            lr=3e-3,
            betas=(0.95,0.95),
            shampoo_beta=None,
            eps=1e-8,
            weight_decay=0.01,
            precondition_frequency=10,
            max_precond_dim=10000
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim
        )
        super().__init__(params, defaults)

    @staticmethod
    def _prj(x: Tensor, QL: Tensor, QR: Tensor) -> Tensor:
        if QL is not None:
            x = QL.T @ x
        if QR is not None:
            x = x @ QR
        return x
    
    @staticmethod
    def _prj_back(x: Tensor, QL: Tensor, QR: Tensor) -> Tensor:
        if QL is not None:
            x = QL @ x
        if QR is not None:
            x = x @ QR.T
        return x

    @staticmethod
    def _eigh_basis(M: Tensor):
        eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        _, Q = torch.linalg.eigh(M + 1e-30 * eye)
        return torch.flip(Q, dims=[1])

    @staticmethod
    def _refresh_basis(M: Tensor, Q: Tensor, exp_avg_sq: Tensor, dim: int):
        est_eig = torch.diag(Q.T @ M @ Q)
        order = torch.argsort(est_eig, descending=True)
        Q = Q[:, order]
        exp_avg_sq = exp_avg_sq.index_select(dim, order)
        Q, _ = torch.linalg.qr(M @ Q)
        return Q, exp_avg_sq

    @torch.no_grad()
    def step(self):
        loss = None
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] is not None else beta2

            p: Tensor
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                is_matrix = p.dim() >= 2
                grad = p.grad.float()
                G = grad.reshape(grad.shape[0], -1) if is_matrix else grad
                state = self.state[p]

                # init
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(G)
                    state["exp_avg_sq"] = torch.zeros_like(G)
                    state["L"] = None
                    state["R"] = None
                    state["QL"] = None
                    state["QR"] = None

                    if is_matrix:
                        m, n = G.shape
                        if m <= group["max_precond_dim"]:
                            state["L"] = (1 - shampoo_beta) * (G @ G.T)
                            state["QL"] = self._eigh_basis(state["L"])
                        if n <= group["max_precond_dim"]:
                            state["R"] = (1 - shampoo_beta) * (G.T @ G)
                            state["QR"] = self._eigh_basis(state["R"])
                    
                    continue

                state["step"] += 1
                t = state["step"]
                QL: Tensor = state["QL"]
                QR: Tensor = state["QR"]
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # Adam

                G_rot = self._prj(G, QL, QR)
                exp_avg.mul_(beta1).add_(G, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)

                # Adam step in eigenbasis

                m_rot = self._prj(exp_avg, QL, QR)
                update_rot = m_rot / (exp_avg_sq.sqrt() + group["eps"])
                update: Tensor = self._prj_back(update_rot, QL, QR)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = group["lr"] * (bias_c2 ** 0.5) / bias_c1

                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-step_size)

                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # Shampoo
                
                if state["L"] is not None:
                    state["L"].mul_(shampoo_beta).add_(G @ G.T, alpha=1 - shampoo_beta)
                if state["R"] is not None:
                    state["R"].mul_(shampoo_beta).add_(G.T @ G, alpha=1 - shampoo_beta)

                if t % group["precondition_frequency"] == 0:
                    if state["L"] is not None:
                        state["QL"], state["exp_avg_sq"] = self._refresh_basis(state["L"], state["QL"], state["exp_avg_sq"], dim=0)

                    if state["R"] is not None:
                        state["QR"], state["exp_avg_sq"] = self._refresh_basis(state["R"], state["QR"], state["exp_avg_sq"], dim=1)

        return loss

    