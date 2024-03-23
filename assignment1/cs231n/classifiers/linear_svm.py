from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # calculate score for each class by multiplying the 
        # input data X[i] with the weights W.
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # for every class
        for j in range(num_classes):
            if j == y[i]:
                # no need to change loss 
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            
            #  If the margin is greater than 0, there is a loss for 
            # this class, so we need to update the loss and gradients.
            if margin > 0:
                loss += margin

                #######################################################
                
                # update gradient for incorrect label
                # This pushes the decision boundary away from this incorrect class.
                dW[:, j] += X[i]   

                # update gradient for correct label 
                # This pulls the decision boundary towards the correct class.
                dW[:, y[i]] -= X[i] 

                #######################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # we added the gradient calculation within the loss update block

    # just like loss, gradient is sum on all training examples, 
    # get the average of it
    dW /= num_train  

    # The regularization term is reg * ||W||^2, where ||W||^2 is 
    # the squared Frobenius norm of the weight matrix W.

    # Taking the derivative of this term with respect to W gives 
    # 2 * reg * W, which is added to dW to ensure that the weights are 
    # penalized and encouraged to stay small during optimization.
    
    # append partial derivative of regularization term
    dW += W * reg * 2

    # By performing these operations, the gradient dW is ready to be 
    # used in an optimization algorithm (e.g., gradient descent) to 
    # update the weights W and minimize the loss function.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

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
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # number of possible labels 
    N = len(y) 

    # score matrix
    # can use np.dot() but if both a and b are 2-D arrays, 
    # using matmul or a @ b is preferred.
    # scores = np.matmul(X, W)
    scores = X @ W  

    # print(scores.shape) # 500 - 10

    # scores for true labels

    # we add a newaxis becuase scores is a 2D matrix
    
    # scores[range(N), y] 
    # This indexing operation selects elements from scores at positions 
    #           (0, y[0]), (1, y[1]), ..., (N-1, y[N-1]).
    
    # scores[range(N), y] -> shape (500,)
    # scores[range(N), y][:, np.newaxis] -> shape (500, 1)
    correct_scores = scores[range(N), y][:, np.newaxis]    
    
    # using the +1 margin
    margins = np.maximum(0, scores - correct_scores + 1)   
    # print(margins.shape) -> (500, 10)

    # sum of all margins divided by all classes except true class
    # with regularized loss addition
    # taking lambda (regularization parameter) as 1
    loss = margins.sum() / N - 1 + reg * np.sum(W**2) # regularized loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # use margins that should contribute to loss
    # converts the boolean matrix to an integer matrix, True -> 1 and False -> 0. 
    dW = (margins > 0).astype(int)    # initial gradient with respect to scores
    
    # dW[range(N), y] selects the elements of dW that correspond to 
    # the correct labels for each example.
    # Subtract the sum of each row of dW from the selected elements.
    dW[range(N), y] -= dW.sum(axis=1) # update gradient to include correct labels
    
    # calculate the gradient with respect to Weights - W
    # multiply transpose of input with gradients
    # average it by dividing to N
    
    # The regularization term is reg * ||W||^2, where ||W||^2 is 
    # the squared Frobenius norm of the weight matrix W.
    dW = X.T @ dW / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW