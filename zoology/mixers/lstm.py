# Adapted from Hippogriff, see LICENCE

import torch
from torch import nn
from torch.nn.functional import softplus, gelu
from accelerated_scan.warp import scan

class qLSTM(nn.Module):
    def __init__(self, d_model=256, d_qk= 100, expansion_factor=2, kernel_size=4, layer_idx: int=None, reversed=True):
        super().__init__()
        dim = d_model
        hidden = int(dim * expansion_factor)
        self.hidden = hidden
        self.input = nn.Linear(dim, 2*hidden, bias=False)
        self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = nn.Linear(hidden, 3*hidden, bias=True)
        self.output = nn.Linear(hidden, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.reversed_sigmoid = reversed

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.gates.weight.normal_(std=hidden**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        _, T, _ = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT

        forget, input, output = self.gates(x).chunk(3, dim=-1)
        if self.reversed_sigmoid:
            alpha = torch.pow(nn.functional.sigmoid(-forget), nn.functional.relu(self.alpha))
        else:
            alpha = forget.sigmoid()
        x = input.sigmoid() * x

        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        h = output.sigmoid() * h
        x = self.output(gelu(gate) * h)
        return x

    def state_size(self, sequence_length: int=2048):
        return 2 * self.hidden