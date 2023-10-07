import torch
from torch import nn

INNER_LAYER_SIZE = 3

class NeuralNetwork(nn.Module):
    '''
    A standard neural network with 3 layers.
        Layer 1: input layer
        Layer 2: hidden layer
        Layer 3: output layer

    This network is sequential and uses ReLU activation functions.

    Note: With Pytorch, the input layer is not explicitly defined.
    '''
    def __init__(self, input_size, output_size, inner_layer_size=INNER_LAYER_SIZE):
        super().__init__()
        self.linear_relu_sequential_model = nn.Sequential(
            nn.Linear(input_size, inner_layer_size),
            nn.ReLU(),
            nn.Linear(inner_layer_size, output_size)
        )

    def forward(self, x):
        logits = self.linear_relu_sequential_model(x)
        return logits