import torch
import torch.nn as nn

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ConvTranspose1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.weight = nn.Parameter(
            torch.randn(( out_channels, in_channels, kernel_size))
        )
        self.bias = nn.Parameter(torch.randn(out_channels))


    def forward(self, x):
        batch_size, in_channels, in_length = x.shape
        out_length = (in_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out = torch.zeros((batch_size, self.out_channels, out_length))
        for i in range(in_length):
            out[..., i:i+self.kernel_size] += torch.einsum("bc,kco->bko", x[..., i], self.weight)

        out += self.bias[None, :, None]
        return out

if __name__ == '__main__':
    conv = ConvTranspose1d(3, 1, 3)
    x = torch.randn((1, 3, 5))
    print(conv(x))