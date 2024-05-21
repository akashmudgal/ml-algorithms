import numpy as np
# Class representing a Univariate Linear Regression ALgorithm
class SimpleLinearRegressor:
    
    def __init__(self, lr: float =0.001,n_iters: int=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(X,self.weights) + self.bias

            dj_dw,dj_db = self.__compute_gradient(X,y)

            self.weights -= self.lr * dj_dw
            self.bias -= self.lr * dj_db

    # compute the gradients dj_dw, dj_db for gradient decent algorithm
    def __compute_gradient(self,X,y):
        n_samples, n_features = X.shape

        y_predicted = self.predict(X)

        dj_dw = (1/n_samples) * np.dot(X.T,(y_predicted - y))

        dj_db = (1/n_samples) * np.sum(y_predicted - y)

        return dj_dw,dj_db
    
    # method to do the prediction based on calculated weights
    def predict(self, X):
        y_predicted = np.dot(X,self.weights) + self.bias

        return y_predicted
