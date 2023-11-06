import sklearn
from sklearn.neural_network import MLPClassifier

INNER_LAYER_SIZE = 512

class NeuralNetwork():
    '''
    A standard neural network with 4 layers.
        Layer 1: input layer
        Layer 2: hidden layer
        Layer 3: hidden layer
        Layer 4: output layer
    
    This network is sequential and uses ReLU activation functions.
    '''
    def __init__(self, learning_rate, epochs, batch_size, inner_layer_size=INNER_LAYER_SIZE):
        self.model = MLPClassifier(
            hidden_layer_sizes=(inner_layer_size, inner_layer_size), 
            activation='relu', 
            solver='sgd', 
            max_iter=epochs,
            alpha=0, # Don't add regularization
            learning_rate_init=learning_rate,
            shuffle=False,
            momentum=0,
            tol=0, # Don't automatically stop upon convergence
            verbose=True
        )

    @classmethod
    def get_model(cls, learning_rate, epochs, batch_size, inner_layer_size=INNER_LAYER_SIZE):
        return cls(learning_rate, epochs, batch_size, inner_layer_size).model