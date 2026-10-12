import torch
from torch import nn


def trainable_parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def gate_mlp(): return nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 3))


class LSF(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.gate = gate_mlp(); self.dim = dim

    def forward(self, q, h_base, pi_hat=None):
        weights = self.gate(h_base.mean(dim=1, keepdim=True)).softmax(1)
        return sum(weights[:, i:i+1] * q[i] for i in range(3)), weights


class RCSF(nn.Module):
    def __init__(self, dim, median=0.0, iqr=1.0):
        super().__init__(); self.gate = gate_mlp(); self.dim = dim
        self.register_buffer("median", torch.tensor(float(median))); self.register_buffer("iqr", torch.tensor(max(float(iqr), 1e-6)))

    def forward(self, q, h_base, pi_hat):
        condition = (torch.log1p(pi_hat.detach()).unsqueeze(1) - self.median) / self.iqr
        weights = self.gate(condition).softmax(1)
        return sum(weights[:, i:i+1] * q[i] for i in range(3)), weights
