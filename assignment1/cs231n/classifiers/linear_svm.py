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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:        
        loss += margin
        dW[:, j] +=  X[i,:]
        dW[:, y[i]] += -X[i,:]

      
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # print("naive loss without regulation:")
  # print(loss)
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  # print(dW)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W)
  correct_class_score = scores[list(range(num_train)), y]
  # print(y)
  # print(scores.shape, len(list(range(num_train))), len(y), correct_class_score.shape)
  correct_class_score = correct_class_score.reshape(num_train, -1)
  # print(correct_class_score.shape)

  margin = scores - correct_class_score + 1
  margin = np.maximum(0, margin)
  margin[np.arange(X.shape[0]), y] = 0
  
  loss = np.sum(margin)/X.shape[0]

  # print("vectorized loss without regulation:")
  # print(loss)
  
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################


  margin[margin > 0] = 1
  margin[np.arange(X.shape[0]), y] = - np.sum(margin, axis =1)
  dW = np.dot(X.T, margin) / X.shape[0] + reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  # print(dW)
  return loss, dW
