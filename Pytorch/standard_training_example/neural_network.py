import torch
from torch import nn

INNER_LAYER_SIZE = 3

class NeuralNetwork(nn.Module):
    '''
    A standard neural network with 3 layers.
        Layer 1: input_size -> INNER_LAYER_SIZE
        Layer 2: INNER_LAYER_SIZE -> INNER_LAYER_SIZE
        Layer 3: INNER_LAYER_SIZE -> ouptut_size

    This network is sequential and uses ReLU activation functions.
    '''
    def __init__(self, input_size, output_size, inner_layer_size=INNER_LAYER_SIZE):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_sequential_model = nn.Sequential(
            nn.Linear(input_size, inner_layer_size),
            nn.ReLU(),
            nn.Linear(inner_layer_size, inner_layer_size),
            nn.ReLU(),
            nn.Linear(inner_layer_size, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_sequential_model(x)
        return logits