import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from helpers import fit, evaluate

# Constants
INPUT_SIZE = 28*28
OUTPUT_SIZE = 10

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 40

# Download the dataset. (NOTE: I'm not writing this one 
#   out specifically because my time will be better spent
#   playing around with hyperparameters and optimizing.)
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Set up dataloaders to load the dataset
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Initialize the model
model = NeuralNetwork(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)

# Initialize the loss function (Here we use the standard cross entropy loss function)
loss_fn = nn.CrossEntropyLoss()

# Initialize the optimizer (Here we use the standard stochastic gradient descent optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
accuracies, losses = fit(model, EPOCHS, train_dataloader, loss_fn, optimizer)

accuracy, loss = evaluate(model, test_dataloader, loss_fn)
print(f"Test Error: Accuracy: {accuracy:>0.1f}% Loss: {loss:>8f}")

# Plot the accuracies and losses
fig, ax1 = plt.subplots()
ax1.plot(accuracies, color="blue")
ax1.set_ylabel("Accuracy", color="blue")
ax1.set_xlabel("Epoch")
ax2 = ax1.twinx()
ax2.plot(losses, color="red")
ax2.set_ylabel("Loss", color="red")
plt.ylim([0, max(plt.ylim())])
fig.tight_layout()
plt.show()