"""
Implementation of Mixture of Experts (MoE) layers and related utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model:int, mlp_ratio:float|int):
        super().__init__()
        self.d_mlp = int(mlp_ratio*d_model)
        self.fc1 = nn.Linear(d_model, self.d_mlp)
        self.fc2 = nn.Linear(self.d_mlp, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    def __init__(self, d_model:int, num_experts:int, mlp_ratio:float|int):
        self.d_mlp = int(mlp_ratio*d_model)
        self.num_experts = num_experts
        self.router = nn.Sequential(
            [
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_experts),
                nn.Softmax(dim=-1)
            ]
        )
        self.experts = nn.ModuleList([MLP(d_model, mlp_ratio) for _ in range(num_experts)])

    def forward(self, x:torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, N, Ch = x.shape
        x_flat = x.reshape(-1, Ch)
        routing_weights = self.router(x_flat)  # (B*N, num_experts)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # (B*N, num_experts, Ch)
        output = torch.einsum('be,bec->bc', routing_weights, expert_outputs)  # (B*N, Ch)
        output = output.reshape(B, N, Ch)
        return output