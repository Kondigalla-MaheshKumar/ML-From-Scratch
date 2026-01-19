import numpy as np

class metrics:
    def mse(self,y_true,y_pred):
        return (np.mean((y_true - y_pred) ** 2))
    
    def mae(self,y_true,y_pred):
        return (np.mean(np.abs(y_true - y_pred)))
    
    def rmse(self,y_true, y_pred):
        return np.sqrt(self.mse(y_true,y_pred))
    
    def r2_score(self,y_true,y_pred):
        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.mean((y_true - np.sum(y_true))**2 )
        return 1 - (ssr/sst)