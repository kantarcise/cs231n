# cs231n

My solutions (and more) for CS231n Assingments in 2022.

Here is [the link](https://cs231n.github.io/assignments2022/assignment1/) for them. Use assignment2 and assignment3 for the other urls.

## Assingment 1

In this assignment you will practice putting together a simple image classification pipeline based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- Understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages).
- Understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- Develop proficiency in writing efficient **vectorized** code with numpy.
- Implement and apply a k-Nearest Neighbor (**kNN**) classifier.
- Implement and apply a Multiclass Support Vector Machine (**SVM**) classifier.
- Implement and apply a **Softmax** classifier.
- Implement and apply a **Two layer neural network** classifier.
- Understand the differences and tradeoffs between these classifiers.
- Get a basic understanding of performance improvements from using **higher-level representations** as opposed to raw pixels, e.g. color histograms, Histogram of Oriented Gradient (HOG) features, etc.

### Q0: numpy-review:
The notebook [`numpy_review.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/numpy_review.ipynb) is there for you to discover some basic usage of Numpy!

### Q1: k-Nearest Neighbor classifier
The notebook [`knn.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/knn.ipynb) will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine
The notebook [`svm.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/svm.ipynb) will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier
The notebook [`softmax.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/softmax.ipynb) will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network
The notebook [`two_layer_net.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/two_layer_net.ipynb) will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features
The notebook [`features.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment1/features.ipynb) will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

## Assingment 2

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- Understand **Neural Networks** and how they are arranged in layered architectures.
- Understand and be able to implement (vectorized) **backpropagation**.
- Implement various **update rules** used to optimize Neural Networks.
- Implement **Batch Normalization** and **Layer Normalization** for training deep networks.
- Implement **Dropout** to regularize networks.
- Understand the architecture of **Convolutional Neural Networks** and get practice with training them.
- Gain experience with a major deep learning framework, such as **TensorFlow** or **PyTorch**.
- Explore various applications of image gradients, including saliency maps, fooling images, class visualizations.

### Q1: Multi-Layer Fully Connected Neural Networks
The notebook [`FullyConnectedNets.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment2/FullyConnectedNets.ipynb) will have you implement fully connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

### Q2: Batch Normalization
In notebook [`BatchNormalization.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment2/BatchNormalization.ipynb) you will implement batch normalization, and use it to train deep fully connected networks.

### Q3: Dropout
The notebook [`Dropout.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment2/Dropout.ipynb) will help you implement dropout and explore its effects on model generalization.

### Q4: Convolutional Neural Networks
In the notebook [`ConvolutionalNetworks.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment2/ConvolutionalNetworks.ipynb) you will implement several new layers that are commonly used in convolutional networks.

### Q5: PyTorch on CIFAR-10
For this part, you will be working with PyTorch, a popular and powerful deep learning framework. Open up [`PyTorch.ipynb`](https://github.com/kantarcise/cs231n/blob/main/assignment2/PyTorch.ipynb). There, you will learn how the framework works, culminating in training a convolutional network of your own design on CIFAR-10 to get the best performance you can.

There was also a Network-Visualization question, but that is moved to Assignment 3.
