from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
		that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	# our loss function is 
	# L_i = - log(  e^yi / sum(e^jk))

	# calculate all scores
	num_train = X.shape[0]
	num_classes = W.shape[1]

	for i in range(num_train):
		# calculate score for each class by multiplying the 
		# input data X[i] with the weights W.
		scores = X[i].dot(W)

		# Prevent numerical instabilities due to 
		# overflow when computing exp().
		f = scores - np.max(scores) 

		# Apply softmax activation function to get probabilities for each class.
		softmax = np.exp(f)/np.sum(np.exp(f))

		# Calculate the cross-entropy loss for the current example.
		loss += -np.log(softmax[y[i]])

		# Update the gradient for each class.
		for j in range(num_classes):
			dW[:,j] += X[i] * softmax[j]
		
		# Subtract the input vector for the correct class
		dW[:,y[i]] -= X[i]

	# calculate average and add regularization for loss
	loss = (loss / num_train) + reg * np.sum(W**2) 

	# calculate average and add regularization for gradients
	dW = (dW / num_train) + 2 * reg * W

	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	# number of possible labels 
	num_train = X.shape[0]	
    
	# score matrix
    # can use np.dot() but if both a and b are 2-D arrays, 
    # using matmul or a @ b is preferred.
    # scores = np.matmul(X, W)
	scores = X @ W  
	
	# scores are normalized by subtracting 
	# the maximum score from each row
	scores = scores - np.max(scores, axis=1, keepdims=True)

	# Softmax Loss
	sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
	softmax_matrix = np.exp(scores)/sum_exp_scores
	
	# add all loss calculated for all scores 
	loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]))
	
	# Weight Gradient
	softmax_matrix[np.arange(num_train), y] -= 1
	dW = X.T @ softmax_matrix
	
	# Average
	loss /= num_train
	dW /= num_train
	
	# add regularization
	loss += reg * np.sum(W * W)
	dW += reg * 2 * W 

	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW
