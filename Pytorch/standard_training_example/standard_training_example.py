import torch
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork
from torch import nn
from helpers import fit, evaluate
from iris_dataset import IrisDataset
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 1000

# Get the dataset
dataset = IrisDataset("./standard_datasets/iris.csv", transform=torch.tensor, target_transform=torch.tensor)
training_data, test_data = dataset.get_from_train_test_splits()

# Set up dataloaders to load the dataset
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Initialize the model
# model = NeuralNetwork(input_size=(28*28), output_size=10, inner_layer_size=512)
model = NeuralNetwork(input_size=4, output_size=3)

# # Initialize the loss function (Here we use the standard cross entropy loss function)
loss_fn = nn.CrossEntropyLoss()

# # Initialize the optimizer (Here we use the standard stochastic gradient descent optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
accuracies, losses = fit(model, EPOCHS, train_dataloader, loss_fn, optimizer)

accuracy, loss = evaluate(model, test_dataloader, loss_fn)
print(f"Test Error: Accuracy: {accuracy:>0.1f}% Loss: {loss:>8f}")

fig, ax1 = plt.subplots()
ax1.plot(accuracies, color="blue")
ax1.set_ylabel("Accuracy", color="blue")
ax1.set_xlabel("Epoch")
ax2 = ax1.twinx()
ax2.plot(losses, color="red")
ax2.set_ylabel("Loss", color="red")
plt.xlabel("Epoch")
plt.ylim([0, max(plt.ylim())])
fig.tight_layout()
plt.show()