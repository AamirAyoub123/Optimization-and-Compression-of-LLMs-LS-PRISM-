import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, layer_U, layer_V):
        super().__init__()
        self.layer_V = layer_V # The first matrix (Input -> Rank)
        self.layer_U = layer_U # The second matrix (Rank -> Output)

    def forward(self, x):
        return self.layer_U(self.layer_V(x))