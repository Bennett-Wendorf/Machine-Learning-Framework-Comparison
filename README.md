# Problem Statement

There are an increasingly large number of machine learning frameworks available that all allow their users to build and train neural networks for any purpose. Each option has its own nuances, strengths, and limitations. With so many possibilities, it is almost certain that new users will have a difficult time deciding what to use and how to use it. This project aims to summarize some of the most popular frameworks and compare and contrast how some standard neural networks can be constructed in each.

# Methodology

## Iris Example

### Dataset
The iris dataset is a common dataset used for classification problems. It contains 150 samples of 3 different species of iris flowers. Each sample has 4 features: sepal length, sepal width, petal length, and petal width. The goal is to classify each sample into one of the 3 species based on the 4 features. The full dataset is available as a module in many machine learning frameworks, or a standardized version can be found in the `standard_datasets` directory. This copy of the dataset can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).

A small sample of the dataset looks like the following:
| Sepal Length | Sepal Width | Petal Length | Petal Width | Species              |
|--------------|-------------|--------------|-------------|----------------------|
| 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa          |
| 4.9          | 3.0         | 1.4          | 0.2         | Iris-setosa          |
| 7.0          | 3.2         | 4.7          | 1.4         | Iris-versicolor      |
| 6.4          | 3.2         | 4.5          | 1.5         | Iris-versicolor      |
| 6.3          | 3.3         | 6.0          | 2.5         | Iris-virginica       |
| 5.8          | 2.7         | 5.1          | 1.9         | Iris-virginica       |

### Network Structure
The network structure used for this example is a simple feed-forward network with 1 hidden layer, making 3 total layers. The input layer has 4 nodes, one for each feature in the dataset. The hidden layer has 3 nodes, and the output layer has 3 nodes, one for each species in the dataset. The network used in this example uses ReLU activations functions, a cross entropy loss function, and stochastic gradient descent as an optimizer. The network structure is visualized below.

![Iris Network Structure](res/Iris_Small_Visualization_11_23_23.png)

## MNIST Example

### Dataset
The MNIST dataset is a common dataset used for image classification problems. It contains 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28x28 pixels, and each pixel is represented by a value between 0 and 255. The goal is to classify each image into one of the 10 digits. The full dataset is available as a module in many machine learning frameworks, which is what was used for these examples. The original dataset can be found in the [MNIST Database](http://yann.lecun.com/exdb/mnist/).

A small sample of the dataset looks like the following:
![MNIST Sample](res/mnist-3.0.1.png)

### Network Structure
The network structure used for this example is a simple feed-forward network with 2 hidden layers, making 4 total layers. The input layer has 784 nodes, one for each pixel in the image. Note that the image is flattened into a 1-dimensional structure before it is inputted into the network. The hidden layers have 512 nodes each, and the output layer has 10 nodes, one for each digit. The network used in this example uses ReLU activations functions, a cross entropy loss function, and stochastic gradient descent as an optimizer. The network structure is visualized below.
![MNIST Network Structure](res/MNIST_Visualization_11_23_23.png)

## Frameworks
The following frameworks were used for this project:
- Pytorch
- Scikit-learn
- Tensorflow/Keras

# Framework Comparison Metrics
- Control over network structure <!-- (I.e. Can I have different activations per layer, options for layer types like conv, etc) -->
- Ease of data import <!-- (Iris examples are probably better for this since I didn't really test with MNIST) -->
- Loss function specification
- Optimizer specification
- Training loop customization
    - Metrics during training 
- Network evaluation
<!-- TODO: Make this a table with networks on score out of 5 -->

# Pytorch <!-- Talk about Pytorch on the comparison metrics, and include any unique details if relevant -->

# Scikit-learn <!-- TODO: Talk about Scikit-learn on the comparison metrics, and include any unique details if relevant -->

# Tensorflow/Keras <!-- TODO: Talk about Tensorflow/Keras on the comparison metrics, and include any unique details if relevant -->

# Discussion <!-- TODO: Talk about what I like and why. Maybe also what I think would be good for beginners -->