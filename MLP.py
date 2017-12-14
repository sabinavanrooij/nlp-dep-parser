"""
Multilayer perceptron to distinguish between head and dependent
"""

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # First linear layer Wx + b with
        # input dim 2 and output dim hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        # Second linear layer Wx + b with
        # input dim hidden_size and output dim 1
        self.h2o = nn.Linear(hidden_size, output_size)
        # The nonlinear function for the hidden layer
        self.tanh = nn.Tanh()
        # The output nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        hidden = self.tanh(self.i2h(input))
        output = self.sigmoid(self.h2o(hidden))
        return output
