import torch
from torch import nn
from torch.nn import BatchNorm1d

class BN1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BN1d, self).__init__()
        self.bn = BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        return self.bn(x)

if __name__=="__main__":
    bn = BN1d(num_features=30)
    x = torch.randn(5, 30)
    y = bn(x)
    print(bn)
    print(sum(p.numel() for p in bn.parameters())) # it should have 2 * 30 = 60 parameters
