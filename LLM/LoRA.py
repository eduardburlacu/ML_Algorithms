"""
LoRA: Low-Rank Adaptation of Large Language Models
Implement LoRA in PyTorch
Reference: https://arxiv.org/abs/2106.09685
"""
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class LoRA(
    nn.Module
):
    def __init__(
            self, 
            model: nn.Module,
            d_in: int,
            d_out: int,
            r: int | None = 8, 
            alpha: float = 1., 
            dropout: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.d_in = d_in
        self.d_out = d_out
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        
    def forward(self, x):
        return self.model(x) + self.dropout(self.alpha * ( self.lora_B(self.lora_A(x))))

if __name__ == "__main__":
    d_in = 3
    d_out = 4
    r = 2
    alpha = 0.04
    model = nn.Linear(d_in, d_out)
    lora = LoRA(model, d_in, d_out, r, alpha)
    print(lora)
    # Example of LoRA usage
    x = torch.randn(1, d_in)
    print(model(x))
    print(lora(x))
    
