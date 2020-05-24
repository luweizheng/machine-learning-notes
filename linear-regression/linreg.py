import numpy as np

class LinearRegression:

    def __init__(self):
        # the weight vector
        self.W = None

    def train(self, X, y, method='bgd', learning_rate=1e-2, num_iters=100, verbose=False):
        """
        Train linear regression using batch gradient descent or stochastic gradient descent

        Parameters
        ----------
        X: training data, shape (num_of_samples x num_of_features), num_of_features rows of training sample, each training sample has num_of_features-dimension features.
        y: target, shape (num_of_samples, 1). 
        method: (string) 'bgd' for Batch Gradient Descent or 'sgd' for Stochastic Gradient Descent
        learning_rate: (float) learning rate or alpha
        num_iters: (integer) number of steps to iterate for optimization
        verbose: (boolean) if True, print out the progress

        Returns
        -------
        losses_history: (list) of losses at each training iteration
        """
        num_of_samples, num_of_features = X.shape

        if self.W is None:
            # initilize weights with values
            # shape (num_of_features, 1)
            self.W = np.random.randn(num_of_features, 1) * 0.001
        losses_history = []

        for i in range(num_iters):

            if method == 'sgd':
                # randomly choose a sample
                idx = np.random.choice(num_of_samples)
                loss, grad = self.loss_and_gradient(X[idx, np.newaxis], y[idx, np.newaxis])
            else:
                loss, grad = self.loss_and_gradient(X, y)
            losses_history.append(loss)

            # Update weights using matrix computing (vectorized)
            self.W -= learning_rate * grad

            if verbose and i % (num_iters / 10) == 0:
                print('iteration %d / %d : loss %f' %(i, num_iters, loss))
        return losses_history


    def predict(self, X):
        """
        Predict value of y using trained weights

        Parameters
        ----------
        X: predict data, shape (num_of_samples x num_of_features), each row is a sample with num_of_features-dimension features.

        Returns
        -------
        pred_ys: (num_of_samples, 1) 1-dimension array of y for num_of_samples samples
        """
        pred_ys = X.dot(self.W)
        return pred_ys


    def loss_and_gradient(self, X, y, vectorized=True):
        """
        Compute the loss and gradients

        Parameters
        ----------
        The same as self.train function

        Returns
        -------
        tuple of two items (loss, gradient)
        loss: (float)
        gradient: (array) with respect to self.W 
        """
        if vectorized:
            return linear_loss_grad_vectorized(self.W, X, y)
        else:
            return linear_loss_grad_for_loop(self.W, X, y)


def linear_loss_grad_vectorized(W, X, y):
    """
    Compute the loss and gradients with weights, vectorized version
    """
    # vectorized implementation 
    num_of_samples = X.shape[0]
    # (num_of_samples, num_of_features) * (num_of_features, 1)
    f_mat = X.dot(W)

    # (num_of_samples, 1) - (num_of_samples, 1)
    diff = f_mat - y 
    loss = 1.0 / 2 * np.sum(diff * diff)
    
    # {(num_of_samples, 1).T dot (num_of_samples, num_of_features)}.T
    gradient = ((diff.T).dot(X)).T

    return (loss, gradient)


def linear_loss_grad_for_loop(W, X, y):
    """
    Compute the loss and gradients with weights, for loop version
    """
    
    # num_of_samples rows of training data
    num_of_samples = X.shape[0]
    
    # num_of_samples columns of features
    num_of_features = X.shape[1]
    
    loss = 0
    
    # shape (num_of_samples, 1) same with W
    gradient = np.zeros_like(W) 
    
    for i in range(num_of_samples):
        X_i = X[i, :] # i-th sample from training data
        f = 0
        for j in range(num_of_features):
            f += X_i[j] * W[j, 0]
        diff = f - y[i, 0]
        loss += np.power(diff, 2)
        for j in range(num_of_features):
            gradient[j, 0] += diff * X_i[j]
            
    loss = 1.0 / 2 * loss

    return (loss, gradient)