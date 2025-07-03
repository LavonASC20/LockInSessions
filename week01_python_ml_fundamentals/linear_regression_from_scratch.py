import numpy as np
from IPython.display import clear_output

class LinearRegression:
    def __init__(self):
        self.beta = None

    
    def loss_func(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray, method: str, lmbda = None) -> float: # helper
        assert method in ['ols', 'ridge', 'lasso']

        if method == 'ols':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0]
        
        elif method == 'ridge':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0] + lmbda/2 * np.linalg.norm(beta[:-1])**2
        
        elif method == 'lasso':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0] + lmbda * np.sum(np.abs(beta[:-1]))


    def gradient_step_func(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray, method: str, lmbda = None) -> np.ndarray: # helper
        assert method in ['ols', 'ridge', 'lasso']

        if method == 'ols':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0]
        
        elif method == 'ridge':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0] + lmbda/y.shape[0] * np.concatenate([beta[:-1], np.array([0])])
        
        elif method == 'lasso':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0] + lmbda/y.shape[0] * np.concatenate([np.sign(beta[:-1]), np.array([0])])
        

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = None, iters: int = None, tol: float = None, method: str = None, lmbda: float = None, batch_size: int = None, random_state: int = None):
        '''
        Parameters: 
        X - Design matrix with no intercept column of ones, contains feature vectors, 
            size [observations x features] in (N, d) form
        y - Target vector, size [observation x 1] in either (N, 1) or (N, ) form
        lr (default = 1e-3) - Learning rate
        iters (default = 1e5) - Max number of iterations 
        tol (default = 1e-6) - Gradient norm convergence tolerance
        method (default = 'ols') - Optimization method in ['ols', 'ridge', 'lasso'] for
                 ordinary least squares, ridge, and lasso, respectively
        lmbda (default = 1) - Regularization factor for ridge and lasso regression
        batch_size (default = N) - Size of mini-batches for mini-batch stochastic gradient descent
        random_state (default = None) - Seed for random data shuffler in stochastic optimizer, reproducibility if needed

        Returns:
        beta - Vector of fitted parameters, size [features, 1] in (d, ) form
        loss - Loss values over each iteration of training
        gradient_norms - Norms of gradients over each iteration of training
        '''
        
        # called function
        if random_state is not None:
            np.random.seed(random_state)
        if batch_size is None or batch_size >= int(X.shape[0]): # if given a batch size, mini-batch SGD is used
            batch_size = int(X.shape[0])
        if lr is None:
            lr = 1e-3
        if iters is None: 
            iters = int(1e4)
        if tol is None:
            tol = 1e-6
        if method is None:
            method = 'ols'
        assert method in ['ols', 'ridge', 'lasso']
        if lmbda is None:
            lmbda = 1.0

        self.X = X
        self.y = y
        self.beta = np.random.rand(self.X.shape[1] + 1, ) # plus 1 to account for bias (intercept) term
        self.lr = abs(lr)
        self.iters = iters
        self.tol = abs(tol)
        self.method = method
        self.lmbda = abs(lmbda)
        self.batch_size = batch_size

        loss = np.zeros((iters, ))
        gradient_norms = np.zeros((iters, ))
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        iter = 0
        while iter < iters:
            # use self.X in below so the matrix doesn't grow in size in each iter
            shuffled_data = np.hstack((X, self.y.reshape(-1, 1))) # mini-batch SGD satisfies observation independence
            np.random.shuffle(shuffled_data)
            X, y = shuffled_data[:, :-1], shuffled_data[:, -1].reshape(-1, )

            for start in range(0, X.shape[0], batch_size):
                if start + batch_size >= X.shape[0]:
                    X_batch = X[start: , :]
                    y_batch = y[start: ]
                else: 
                    X_batch = X[start: start + batch_size, :]
                    y_batch = y[start: start + batch_size]
                
                loss[iter] += y_batch.shape[0] * self.loss_func(X_batch, y_batch, self.beta, method, self.lmbda)
                grad = self.gradient_step_func(X_batch, y_batch, self.beta, method, self.lmbda)
                gradient_norms[iter] += np.linalg.norm(grad)
                self.beta -= lr * grad
            loss[iter] /= y.shape[0]
            if np.allclose(gradient_norms[iter], 0, tol):
                print(f"Gradient stabilized in iteration {iter}")
                break
            if iter % 1000 == 0:
                print(f'Iteration: {iter} \nLoss: {loss[iter]} \nGradient Norm: {gradient_norms[iter]}')
            if iter % 3000 == 0:
                clear_output(wait = True)


            iter += 1

        print(f'Fit complete in iteration {iter}')

        return self.beta, loss, gradient_norms
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Parameters:
        x - Batch of any number of observations , i.e. of size [any x d], from which to make predictions

        Returns:
        np.ndarray containing predictions for each observation in the batch, size [any x 1] 
        '''

        if self.beta is None:
            raise ValueError('Model must be fit to data first, call the fit() method before prediction')
        return np.hstack((x, np.ones((x.shape[0], 1)))).dot(self.beta)
    
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Parameters:
        x - Batch of any number of observations , i.e. of size [any x d], from which to score model predictions
        y - Batch of targets associated to observations in x, size [any x 1] from which to score model predictions

        Returns:
        float of MSE score of predictions across the batch given in x
        '''

        return np.mean((y - self.predict(x))**2)
    