import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01

matplotlib.rcParams['figure.figsize'] = [9,6]

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

def f(x):
    return x**2 + 2*x - 5

y = f(x) + tf.random.normal(shape=[201])

class Model(tf.Module):
    def __init__(self):
        # Randomly generate weight and bias terms
        rand_init = tf.random.uniform(shape=[3], minval=0, maxval=5, seed=22)
        # Initialize model params
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])
    
    @tf.function
    def __call__(self, x):
        # Quadratic model: quadratic_weight * x^2 + linear_weight * x + bias
        return self.w_q * x**2 + self.w_l * x + self.b

quad_model = Model()

def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, f(x), label='Ground Truth')
    plt.plot(x, model(x), label='Predictions')
    plt.legend()
    plt.title(title)
    plt.show()

plot_preds(x, y, f, quad_model, 'Untrained Model')

def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(BATCH_SIZE)

losses = []

# Format training loop
for epoch in range(EPOCHS):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = quad_model(x_batch)
            batch_loss = mse_loss(y_pred, y_batch)
        # Update params with respect to the gradient calculations
        gradients = tape.gradient(batch_loss, quad_model.variables)
        for g, v in zip(gradients, quad_model.variables):
            v.assign_sub(g * LEARNING_RATE)
    loss = mse_loss(quad_model(x), y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.numpy():0.3f}')

# Plot model results
print("\n")
plt.plot(range(EPOCHS), losses)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE loss vs training iterations')

plot_preds(x, y, f, quad_model, 'Trained Model')