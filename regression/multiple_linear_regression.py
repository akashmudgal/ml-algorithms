import numpy as np

class MultipleLinearRegressor:
    
    def __init__(self, lr: float =0.001,n_iters: int=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def predict(self, X):
        n_samples = X.shape[0]
        y_predicted = np.zeros(n_samples)
        
        for i in range(n_samples):
            y_predicted[i] = np.dot(self.weights,X[i]) + self.bias
        return y_predicted
    
    def compute_cost(self, X, y):
        # set the cost variable to zero initially
        cost = 0
        
        # number of training examples
        n_samples = X.shape[0]

        y_predicted =self.predict(X)
        
        cost = np.sum((y_predicted - y) ** 2) / (2*n_samples)

        return cost
    
    def __compute_gradient(self, X, y):
        y_predicted = self.predict(X)

        n_samples = X.shape[0]

        dj_dw = np.dot(X.T,y_predicted - y) / n_samples
        dj_db = np.mean(y_predicted - y)

        return dj_dw,dj_db
    
    def fit(self,X,y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            dj_dw,dj_db = self.__compute_gradient(X,y)
            self.weights -= self.lr * dj_dw
            self.bias -= self.lr * dj_db