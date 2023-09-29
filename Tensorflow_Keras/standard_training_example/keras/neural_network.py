import tensorflow as tf

INNER_LAYER_SIZE = 3

class NeuralNetwork():
    '''
    A standard neural network with 3 layers.
        Layer 1: input_size -> INNER_LAYER_SIZE
        Layer 2: INNER_LAYER_SIZE -> INNER_LAYER_SIZE
        Layer 3: INNER_LAYER_SIZE -> ouptut_size

    This network is sequential and uses ReLU activation functions.
    '''
    def __init__(self, input_size, output_size, inner_layer_size=INNER_LAYER_SIZE):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=input_size, activation='relu'),
            tf.keras.layers.Dense(units=inner_layer_size, activation='relu'),
            tf.keras.layers.Dense(units=output_size, activation='softmax')
        ])

    @classmethod
    def get_model(cls, input_size, output_size, inner_layer_size=INNER_LAYER_SIZE):
        return cls(input_size, output_size, inner_layer_size).model