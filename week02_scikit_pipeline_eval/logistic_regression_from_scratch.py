import numpy as np
from IPython.display import clear_output

class LogisticRegression():
    def __init__(self, lr: float = 1e-3, iters: int = 15000, tol: float = 1e-6, 
                 lmbda: float = 0.0, reg: str = 'none', batch_size: int = None, 
                 verbose: bool = True, random_state: int = 611):
        self.beta = None
        self.lr = lr
        self.iters = iters
        self.tol = tol
        self.lmbda = lmbda
        self.reg = reg
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def bce_loss(self, y_true, y_pred):
        assert self.reg in ['none', 'ridge', 'lasso']
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # clip to prevent instability in logarithm calculations
        bce = -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        if self.reg == 'ridge':
            bce += self.lmbda/(2*len(y_true)) * np.sum((self.beta[:-1])**2)
        elif self.reg == 'lasso':
            bce += self.lmbda/len(y_true) * np.sum(np.abs(self.beta[:-1]))
        return bce

    def bce_gradient(self, X, y_true, y_pred):
        assert self.reg in ['none', 'ridge', 'lasso']
        bce_grad = -np.mean((y_true - y_pred)[:, np.newaxis] *X, axis = 0)
        if self.reg == 'ridge':
            bce_grad += self.lmbda * np.concatenate([self.beta[:-1], np.array([0])])
        elif self.reg == 'lasso':
            bce_grad += self.lmbda * np.concatenate([np.sign(self.beta[:-1]), np.array([0])])
        return bce_grad
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.y = y
        self.beta = np.random.rand(self.X.shape[1] + 1, )
        if self.batch_size == None:
            self.batch_size = self.X.shape[0]

        loss = np.zeros((self.iters, ))
        gradient_norms = np.zeros((self.iters, ))
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        iter = 0
        while iter < self.iters:
            shuffled_data = np.hstack((X, y.reshape(-1,1)))
            np.random.shuffle(shuffled_data)
            X, y = shuffled_data[:, :-1], shuffled_data[:, -1].reshape(-1, )
            for start in range(0, X.shape[0], self.batch_size):
                if start + self.batch_size >= X.shape[0]:
                    X_batch, y_batch = X[start: , :], y[start:]
                else:
                    X_batch, y_batch = X[start: start + self.batch_size, :], y[start: start + self.batch_size]
                loss[iter] += X_batch.shape[0] * self.bce_loss(y_batch, self.sigmoid(X_batch @ self.beta))
                grad = self.bce_gradient(X_batch, y_batch, self.sigmoid(X_batch @ self.beta))
                gradient_norms[iter] += np.linalg.norm(grad)
                self.beta -= self.lr * grad
            loss[iter] /= X.shape[0]

            if self.verbose == True:
                if gradient_norms[iter] <= self.tol:
                    print(f'Gradient converged to 0 in {iter} iterations')
                    break
                if iter % 1000 == 0:
                    print(f'Iteration: {iter} \nLoss: {loss[iter]:.2f} \nGradient Norm: {gradient_norms[iter]:.2f}\n\n')
                if iter % 3000 == 0:
                    clear_output(wait = True)

            iter += 1

        print(f'Fit complete in iteration {iter}')
        return self.beta, loss, gradient_norms
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int) # deterministic for training and inference (optimality), 
                                                          # can consider vectorized sampling for 
                                                          # generative modeling cases though                                            

    def predict_proba(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        probs = self.sigmoid(X @ self.beta)
        return probs

    def score(self, X, y): # accuracy score function
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.mean(self.predict(X) == y)

