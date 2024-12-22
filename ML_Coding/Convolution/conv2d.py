from math import floor
import torch
import torch.nn as nn
from torch import einsum

class Conv2d(nn.Module):
    """
    Input shape: (batch_size, in_channels, in_length)
    Output shape: (batch_size, out_channels, out_length)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, device, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else: # tuple/list of (int, int)
            assert isinstance(kernel_size, (tuple, list)), f"Expected tuple or list, got {type(kernel_size)}"
            assert len(kernel_size) == 2, f"Expected 2 elements, got {len(kernel_size)}"
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        else: # tuple/list of (int, int)
            assert isinstance(stride, (tuple, list)), f"Expected tuple or list, got {type(stride)}"
            assert len(stride) == 2, f"Expected 2 elements, got {len(stride)}"
            self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        else: # tuple/list of (int, int)
            assert isinstance(padding, (tuple, list)), f"Expected tuple or list, got {type(padding)}"
            assert len(padding) == 2, f"Expected 2 elements, got {len(padding)}"
            self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        else: # tuple/list of (int, int)
            assert isinstance(dilation, (tuple, list)), f"Expected tuple or list, got {type(dilation)}"
            assert len(dilation) == 2, f"Expected 2 elements, got {len(dilation)}"
            self.dilation = dilation
        self.dilated_kernel_size = [None, None]
        for i in range(2):
            self.dilated_kernel_size[i] = 1 + dilation[i] * (kernel_size[i] - 1)

        self.device = device
        self.dtype = dtype

        self.weights = nn.UninitializedParameter()
        self.bias = nn.UninitializedParameter()
        self.init_weights()

    def pad_sequence(self, x: torch.Tensor):
        pad = [None, None]
        pad[0] = torch.zeros(
            x.shape[0], self.in_channels, self.padding[0], 1,
            device=self.device, dtype=self.dtype
        )
        pad[1] = torch.zeros(
            x.shape[0], self.in_channels, 1, self.padding[1],
            device=self.device, dtype=self.dtype
        )
        x = torch.cat([pad[0], x, pad[0]], dim=-2)
        x = torch.cat([pad[1], x, pad[1]], dim=-1)
        return x

    def init_weights(self):
        self.weights = nn.Parameter(
            torch.randn(
                (self.out_channels, self.in_channels, *self.kernel_size),
                dtype=self.dtype,
                device=self.device
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(
                self.out_channels,
                dtype=self.dtype,
                device=self.device
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape: (batch_size, in_channels, in_length)
        Output shape: (batch_size, out_channels, out_length)
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.pad_sequence(x)
        bsz, in_channels, in_length = x.shape
        assert in_channels == self.in_channels, f"Expected {self.in_channels} channels, got {in_channels}"
        out_length = floor(1+(in_length-self.dilated_kernel_size)/self.stride)
        out_shape = (bsz, self.out_channels, out_length)
        out = torch.zeros(out_shape, device=self.device, dtype=self.dtype)
        for i in range(out_length):
            out[...,i] =  self.bias[None,:] + einsum(
                "bit,oit->bo",
                x[..., i*self.stride:i*self.stride+self.dilated_kernel_size:self.dilation],
                self.weights
            )
        return out

if __name__ == "__main__":
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    conv = Conv1d(3, 2, 3, 2, 1, 2, device, dtype)
    x = torch.ones((2, 3, 8), device=device, dtype=dtype)
    y = conv(x)
    print(y.shape)
    print(y)