import numpy as np

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
        

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float, iters: int, tol: float, method: str = None, lmbda: float = None, batch_size: int = None):
        if batch_size is None or batch_size >= int(X.shape[0]): # if given a batch size, mini-batch SGD is used
            batch_size = X.shape[0]
        if lr is None:
            lr = 10e-3
        if iters is None: 
            iters = 10e3
        if tol is None:
            tol = 10e-6
        if method is None:
            method = 'ols'
        assert method in ['ols', 'ridge', 'lasso']
        if lmbda is None and method in ['ridge', 'lasso']:
            lmbda = 1.0

        self.X = X
        self.y = y
        self.beta = np.random.rand(self.X.shape[1], )
        self.lr = abs(lr)
        self.iters = iters
        self.tol = abs(tol)
        self.method = method
        self.lmbda = abs(lmbda)
        self.batch_size = batch_size
        
        shuffled_data = np.hstack((X, y.reshape(-1, 1))) # mini-batch SGD satisfies observation independence
        np.random.shuffle(shuffled_data)
        X, y = shuffled_data[:, :-1], shuffled_data[:, -1].reshape(-1, )
        X = np.hstack((X, np.ones(X.shape[0])))

        loss = np.zeros((iters, ))
        gradient_norms = np.zeros((iters, ))

        for iter in range(iters):
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
                break
            if iter % 10 == 0:
                print(f'Iter: {iter} \nLoss: {loss[-1]} \nGradient Norm: {gradient_norms[-1]}')
            
        print('fit complete')

        return self.beta, loss, gradient_norms
    
    def predict(self, x: np.ndarray):
        if self.beta is None:
            raise ValueError('Model must be fit to data first, call the fit() method before prediction')
        return np.hstack((x, np.ones(x.shape[0]))).dot(self.beta)
    
    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean((y - self.predict(x))**2)
    