import tensorflow as tf
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork
from iris_dataset import IrisDataset

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
model = NeuralNetwork.get_model(INPUT_SIZE, OUTPUT_SIZE)

# Initialize the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Initialize the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# Evaluate the model
loss, accuracy = model.evaluate(
    x=X_test,
    y=y_test,
    verbose=0
)
print(f"Test Error: Accuracy: {(accuracy*100):>0.1f}% Loss: {loss:>8f}")

# Plot the results
fig, ax1 = plt.subplots()
ax1.plot(history.history['sparse_categorical_accuracy'], color="blue")
ax1.set_ylabel("Accuracy", color="blue")
ax1.set_xlabel("Epoch")
ax2 = ax1.twinx()
ax2.plot(history.history['loss'], color="red")
ax2.set_ylabel("Loss", color="red")
plt.ylim([0, max(plt.ylim())])
fig.tight_layout()
plt.show()