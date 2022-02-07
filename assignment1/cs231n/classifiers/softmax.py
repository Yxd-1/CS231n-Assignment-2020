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
    '''
    for i in range(X.shape[0]):
      score = np.dot(X[i], W)
      score -= max(score)   # 减去最大值，防止score过大取指数发生溢出
      score = np.exp(score)
      softmax_sum = np.sum(score)   #算分母
      score /= softmax_sum    # 得到softmax的概率分布
      # 计算梯度
      for j in range(W.shape[1]):
        if j != y[i]:
          dW[:, j] += score[j] * X[i]
        else:
          dW[:, j] -= (1 - score[j]) * X[i]
      loss -= np.log(score[y[i]])
    loss /= X.shape[0]    # 平均
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)   #正则化系数
    dW += 2 * reg * W     #正则化项求导后的值
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        sum_scores = np.sum(np.exp(scores))
        loss -= scores[y[i]]
        loss += np.log(sum_scores)
        for j in range(num_classes):
            dW[:, j] += X[i] * np.exp(scores[j]) / sum_scores
            if j == y[i]:
                dW[:, j] -= X[i]

    dW /= num_train
    dW += reg * W
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

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
    '''
    scores = np.dot(X, W)   #计算得分
    scores -= np.max(scores, axis=1, keepdims=True)   #数值稳定性
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1, keepdims=True)   #计算softmax
    ds = np.copy(scores)    #初始化loss对scores的梯度
    ds[np.arange(X.shape[0]), y] -= 1   #求出scores的梯度
    dW = np.dot(X.T, ds)    #求出W的梯度
    loss += scores[np.arange(X.shape[0]), y]   #计算loss
    loss += -np.log(loss).sum()    #求交叉熵
    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1,keepdims=True)    # 减去最大值，稳定数值
    sum_scores = np.sum(np.exp(scores), 1)
    loss -= np.sum(scores[np.arange(num_train), y])
    loss += np.sum(np.log(sum_scores))

    ret = np.zeros(scores.shape)
    ret += np.exp(scores) / sum_scores.reshape(-1, 1)
    ret[range(num_train), y] -= 1

    dW += X.T.dot(ret)

    dW /= num_train
    dW += reg * W
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
