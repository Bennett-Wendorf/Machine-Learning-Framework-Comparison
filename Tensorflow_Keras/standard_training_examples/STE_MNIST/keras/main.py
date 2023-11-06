import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork

# Constants
INPUT_SIZE = 28*28
OUTPUT_SIZE = 10

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 40

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

# Reshape the data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Initialize the model
model = NeuralNetwork.get_model(INPUT_SIZE, OUTPUT_SIZE)

# Initialize the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Initialize the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# Evaluate the model
loss, accuracy = model.evaluate(
    x=x_test,
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
