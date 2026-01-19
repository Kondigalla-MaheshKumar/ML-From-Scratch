import numpy as np

class LinearRegression:
    def __init__(self,learning_rate = 0.01,epochs = 1000,optimizer = "BGD",batch_size = None,shuffle = True,early_stop = False,tol= 1e-8):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer.upper()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.early_stop = early_stop
        self.tol = tol

        self.weights = None
        self.bias = None
        self.loss_history = []

    def _gradients(self,X,y):
        m = X.shape[0]
        y_pred = np.dot(X, self.weights) + self.bias
        error = y - y_pred

        dw = - (1 / m ) * np.dot(X.T,error)
        db = - (1 / m ) * np.sum(error)
        return dw,db
    def _loss(self,y,y_pred):
        m = len(y)
        loss = ( 1/ ( 2 * m ) )* np.sum((y - y_pred) ** 2)
        return loss
    def fit(self,X,y):
        if X.ndim == 1:
            X = X.reshape(-1,1)

        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)
        self.bias = 0.0

        valid_optimizers = {"BGD","SGD", "MBGD"}
        if self.optimizer not in valid_optimizers:
            raise ValueError("Optimizer must be in BGD,SGD, MBGD")
        if self.optimizer == "MBGD" and self.batch_size is None:
            raise ValueError("batch size must be give when optimizer == MBGD ")
        
        X_full = X.copy()
        y_full = y.copy()
        prev_loss = float("inf")

        for epoch in range(self.epochs):
            if self.shuffle and self.optimizer in {"SGD", "MBGD"}:
                index = np.random.permutation(n_samples)
                X,y = X_full[index], y_full[index]
            else: 
                X,y = X_full, y_full
            if self.optimizer == "BGD":
                dw,db = self._gradients(X,y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            elif self.optimizer == "SGD":
                for index in range(n_samples):
                    X_batch = X[index: index + 1]
                    y_batch = y[index: index + 1]

                    dw,db = self._gradients(X_batch,y_batch)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            elif self.optimizer == "MBGD":
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X[start : end]
                    y_batch = y[start : end]

                    dw,db = self._gradients(X_batch,y_batch)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db   

            y_pred = np.dot(X, self.weights) + self.bias
            loss = self._loss(y,y_pred)
            self.loss_history.append(loss)

            if epoch % 100 ==0:
                print(f"Epoch : {epoch} Loss : {loss: .6f}")
            if self.early_stop and abs(prev_loss - loss) < self.tol:
                print(f"Early stopping Epoch {epoch}")
                break
            prev_loss = loss
        return self.weights, self.bias
    def predict(self,X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return np.dot(X,self.weights) + self.bias
