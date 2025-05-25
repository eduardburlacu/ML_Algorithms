from typing import Tuple

import torch
import torch.utils.cpp_extension

BITS = 8
QMAX = 2**BITS - 1

quantizer = torch.utils.cpp_extension.load(
    name="quantizer",
    sources = ["LLM/quantization.cu"],
    extra_cuda_cflags=["--use_fast_math"],
    verbose=True
)

def quantize_weights_8bit(weights: torch.Tensor)->Tuple[torch.Tensor, float]:
    assert weights.is_cuda
    w_max = weights.abs().max()
    scale = w_max.item() / QMAX if w_max.item() != 0 else 1.0
    weights_q = torch.empty_like(weights, dtype=torch.int8)
    quantizer.quantize_weights_cuda(
        weights.contiguous(),
        weights_q,
        scale
    )
    return  weights_q, scale

if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(1024, 512).cuda()
    W_q, scale = quantize_weights_8bit(W)
    print("Scale:", scale)
    print("Quantized weights dtype:", W_q.dtype)
    print("First few values:", W_q.view(-1)[:10])
