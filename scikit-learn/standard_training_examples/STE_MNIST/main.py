import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork

# Constants
INPUT_SIZE = 28*28
OUTPUT_SIZE = 10

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 40

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', cache=True)

x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2)

# Initialize the model
model = NeuralNetwork.get_model(LEARNING_RATE, EPOCHS, BATCH_SIZE)

# Train the model
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
print(f"Test Error: Accuracy: {(metrics.accuracy_score(y_test, y_pred)*100)}%")

# Plot the results
# NOTE: scikit-learn doesn't appear to have a way to track accuracy over time
plt.plot(model.loss_curve_, color="red")
plt.ylabel("Loss", color="red")
plt.xlabel("Epoch")
plt.ylim([0, max(plt.ylim())])
plt.show()