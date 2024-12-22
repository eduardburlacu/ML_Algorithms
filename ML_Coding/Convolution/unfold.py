"""
Unfold operation

Extracts sliding local blocks from a batched input tensor.
"""
import torch
from torch import nn
import torch.nn.functional as F

def api_test_1():
    unfold = nn.Unfold(
        kernel_size=(2,3)
    )
    x = torch.randn(3, 5, 3, 4)
    y = unfold(x)
    """Clarification:
    bsz is unchanged
    each of (s1, s2) is patchified, resulting in a total count of (1 + size - k + 2 * pad)/stride per dimension
    """
    print(y.shape)
    return y

def api_test_2():
    """
    Convolution is equivalent to Unfold + Matrix Multiplication + Fold
    """
    x = torch.randn(1, 3, 10, 12)
    w = torch.randn(2, 3, 4, 5)
    x_unf = F.unfold(x, kernel_size=(4, 5), padding=0, stride=1)
    y_unf = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    y = F.fold(y_unf, (7, 8), (1, 1))
    return y


if __name__=="__main__":
    api_test_1()
    api_test_2()
