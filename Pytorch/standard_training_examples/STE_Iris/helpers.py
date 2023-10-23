import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import nn
from torch.nn.modules.loss import _Loss
from typing import Callable
from typing import List, Tuple

def fit(model: nn.Module, epochs: int, dataloader: DataLoader, loss_fn: _Loss, optimizer: Optimizer) -> Tuple[List[float], List[float]]:
    accuracy, loss = [], []
    for epoch in range(epochs):
        epoch_accuracy, epoch_loss = _epoch_train(dataloader, model, loss_fn, optimizer)
        accuracy.append(epoch_accuracy)
        loss.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Accuracy: {epoch_accuracy:>0.1f}%, Loss: {epoch_loss:>8f}")
    return accuracy, loss

def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss) -> Tuple[float, float]:
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary for this example, but good practice
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, num_correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations or memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = 100*(num_correct / size)
    return accuracy, test_loss

def _epoch_train(dataloader: DataLoader, model: nn.Module, loss_fn: _Loss, optimizer: Optimizer) -> Tuple[float, float]:
    epoch_loss, num_correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # Compute gradients
        optimizer.step() # Update parameters
        optimizer.zero_grad() # Set gradients to zero

        epoch_loss += loss.item()
        num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    accuracy = 100*(num_correct / len(dataloader.dataset))
    epoch_loss /= len(dataloader)

    return accuracy, epoch_loss