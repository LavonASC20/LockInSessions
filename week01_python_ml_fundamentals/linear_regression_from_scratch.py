import numpy as np
from IPython.display import clear_output

class LinearRegression:
    def __init__(self):
        self.beta = None

    def loss_func(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray, method: str, lmbda = None) -> float:
        assert method in ['ols', 'ridge', 'lasso']

        if method == 'ols':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0]
        
        elif method == 'ridge':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0] + lmbda * np.linalg.norm(beta[:-1])**2/beta[:-1].shape[0]
        
        elif method == 'lasso':
            return (y-X.dot(beta)).T.dot(y-X.dot(beta))/y.shape[0] + lmbda * np.sum(np.abs(beta[:-1]))/beta[:-1].shape[0]


    def gradient_step_func(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray, method: str, lmbda = None) -> np.ndarray:
        assert method in ['ols', 'ridge', 'lasso']

        if method == 'ols':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0]
        
        elif method == 'ridge':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0] + lmbda * 2*np.concatenate([beta[:-1], np.array([0])])/beta.shape[0]
        
        elif method == 'lasso':
            return -2*X.T.dot(y-X.dot(beta))/y.shape[0] + lmbda * np.concatenate([np.sign(beta[:-1]), np.array([0])])/beta.shape[0]
        

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = None, iters: int = None, tol: float = None, method: str = None, lmbda: float = None, batch_size: int = None):
        if batch_size is None or batch_size >= int(X.shape[0]): # if given a batch size, mini-batch SGD is used
            batch_size = int(X.shape[0])
        if lr is None:
            lr = 1e-3
        if iters is None: 
            iters = int(1e5)
        if tol is None:
            tol = 1e-6
        if method is None:
            method = 'ols'
        assert method in ['ols', 'ridge', 'lasso']
        if lmbda is None:
            lmbda = 1.0

        self.X = X
        self.y = y
        self.beta = np.random.rand(self.X.shape[1] + 1, ) # plus 1 to account for bias term
        self.lr = abs(lr)
        self.iters = iters
        self.tol = abs(tol)
        self.method = method
        self.lmbda = abs(lmbda)
        self.batch_size = batch_size

        loss = np.zeros((iters, ))
        gradient_norms = np.zeros((iters, ))

        iter = 0
        while iter < iters:
            # use self.X in below so the matrix doesn't grow in size in each iter
            shuffled_data = np.hstack((self.X, self.y.reshape(-1, 1))) # mini-batch SGD satisfies observation independence
            np.random.shuffle(shuffled_data)
            X, y = shuffled_data[:, :-1], shuffled_data[:, -1].reshape(-1, )
            X = np.hstack((X, np.ones((X.shape[0], 1))))

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
                print(f'Iter: {iter} \nLoss: {loss[iter]} \nGradient Norm: {gradient_norms[iter]}')
            if iter % 3000 == 0:
                clear_output(wait = True)


            iter += 1

        print(f'Fit complete in iteration {iter}')

        return self.beta, loss, gradient_norms
    
    def predict(self, x: np.ndarray):
        if self.beta is None:
            raise ValueError('Model must be fit to data first, call the fit() method before prediction')
        return np.hstack((x, np.ones((x.shape[0], 1)))).dot(self.beta)
    
    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean((y - self.predict(x))**2)
    