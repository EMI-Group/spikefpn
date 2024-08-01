import torch.nn as nn
from .spike_neurons import *
from collections import namedtuple

Genotype = namedtuple("Genotype_2D", "cell cell_concat")

PRIMITIVES = [
    "skip_connect",
    "snn_b3",
    "snn_b5"
]

OPS = {
    "snn_b3": lambda Cin,Cout, stride, signal: SNN_2d(Cin, Cout, kernel_size=3, stride=stride,b=3), 
    "snn_b5": lambda Cin,Cout, stride, signal: SNN_2d(Cin, Cout, kernel_size=3, stride=stride,b=5)
}


class Identity(nn.Module):
    def __init__(self, C_in, C_out, signal):
        super(Identity, self).__init__()
        self._initialize_weights()
        self.conv1 = nn.Conv2d(C_in,C_out,1,1,0)
        self.signal = signal

    def forward(self, x):
        if self.signal:
            return self.conv1(x)
        else:
            return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
