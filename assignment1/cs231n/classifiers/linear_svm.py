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
    dW = np.zeros(W.shape)  # initialize the gradient as zero  W.shape=(3073,10) X.shape=(500,3073)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0    # scores是计算出来第i个样本对应每一类的分数，correct里存放第i个样本所属分类的分数
    for i in range(num_train):
        scores = X[i].dot(W)    # 向量内积，多维矩阵乘法 (1,3073)*(3073,10)=(1,10)
        correct_class_score = scores[y[i]]   # y[i] provides index of the label of x[i]
        for j in range(num_classes):
            if j == y[i]:
                continue        # 自己不减自己的得分
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:          # 当别的类的得分比所属分类的得分高一分以上时,margin大于0，即loss大于0
                loss += margin
                dW[:, j] += X[i].T     # 每一个类里加上X[i].T,再在X[i]对应的类y[i]里减去X[i].T
                dW[:, y[i]] -= X[i].T     # 除了X[i]对应的类外其他的类都加上X[i].T,X[i]对应的类多次减去X[i].T
                # 加上X[i]和减去X[i]是因为数学上的推导结果是这个
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.正则化项
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)       # N x C
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)  #(N,1),正确分类的分数
    margins = np.maximum(0, scores - correct_class_scores + 1)      # 找最大值
    margins[range(num_train), list(y)] = 0     # 将正确分类的赋值为0,这一步是将margins里对应元素赋值为0
    loss += margins.sum() / num_train        # loss参照之前的公式即可
    loss += 0.5 * reg * np.sum(W * W)       # 正则化项

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

    coeff_mat = np.zeros((num_train,num_classes))   # 求梯度，coeff_mat是L对s的导数
    coeff_mat[margins > 0] = 1      # 此处赋值参考markdown的公式
    coeff_mat[range(num_train),list(y)] = 0
    coeff_mat[range(num_train),list(y)] = -np.sum(coeff_mat, axis = 1)  # 对正确的项导数减去若干个xi

    dW += np.dot(X.T, coeff_mat) 
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
