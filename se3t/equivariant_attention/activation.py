import torch
from torch import nn

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(torch.nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        sigmoid = (1 + (-self.beta * x).exp()).pow(-1)
        return x * sigmoid


class SwishBeta(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwishBeta, self).__init__()
        self.beta = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.ones_(self.beta.weight)

    def forward(self, x):
        sigmoid = (1 + (-self.beta(x)).exp()).pow(-1)
        return x * sigmoid
        
