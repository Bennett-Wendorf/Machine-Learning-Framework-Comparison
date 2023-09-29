import matplotlib.pyplot as plt
from sklearn import metrics
from iris_dataset import IrisDataset
from neural_network import NeuralNetwork

# Constants
INPUT_SIZE = 4
OUTPUT_SIZE = 3

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 1000

dataset = IrisDataset("./standard_datasets/iris.csv")
(X_train, y_train), (X_test, y_test) = dataset.get_from_train_test_splits()

# Initialize the model
model = NeuralNetwork.get_model(LEARNING_RATE, EPOCHS, BATCH_SIZE)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Test Error: Accuracy: {(metrics.accuracy_score(y_test, y_pred)*100)}%")

# Plot the results
# NOTE: scikit-learn doesn't appear to have a way to track accuracy over time
plt.plot(model.loss_curve_, color="red")
plt.ylabel("Loss", color="red")
plt.xlabel("Epoch")
plt.ylim([0, max(plt.ylim())])
plt.show()
