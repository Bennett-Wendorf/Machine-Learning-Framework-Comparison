import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

def f(x):
    return x**2 + 2*x - 5

y = f(x) + tf.random.normal(shape=[201])

def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, f(x), label='Ground Truth')
    plt.plot(x, model(x), label='Predictions')
    plt.legend()
    plt.title(title)
    plt.show()

new_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.stack([x, x**2], axis=1)),
    tf.keras.layers.Dense(units=1, kernel_initializer=tf.random.normal)
])

new_model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
)

history = new_model.fit(x, y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0)

new_model.save('./keras_model.keras')

plt.plot(history.history['loss'])
plt.title('Keras training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([0, max(plt.ylim())])

plot_preds(x, y, f, new_model, 'Keras Model')