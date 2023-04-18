
"""
    Naive Classificator - In future it will only House Classificator and Calls
    For now it is just established model based on Altitude as well as  Wilderness Type

    Some Values have been detemined from info file and some have been established via own search

"""
import pandas as pd 
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

#Determine which 

class NaiveCoverClassifer(BaseEstimator, ClassifierMixin):

    def __init__(self,**kwargs):
        """
            data are used in already preprocessed 
        """
        self.kwargs = kwargs
        

    def fit(self, X, y=None):
        # Just preprocess the X data accordingly

        pass
    
    def predict(self,X, y=None):
        X_temp = X.copy()
        X_temp = X_temp[:,0:14]
        dataInt = pd.DataFrame(data=X_temp,columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])

        dataInt['PredictedCoverage'] = np.zeros(len(dataInt))
        #  PRedicitons for 2 wilderness types
        dataInt.loc[(dataInt[11] == 1),'PredictedCoverage'] = 1

        # Predictions for 1 wilderness type 
        # To Naively discriminate in this wilderness type between 2 and 1 is chalenging 
        dataInt.loc[(dataInt[10] == 1) & (dataInt[0] > 3333),'PredictedCoverage'] = 1
        dataInt.loc[(dataInt[10] == 1) & (dataInt[0] <= 3333),'PredictedCoverage'] = 2
        
        # Predictions for 3 wilderness type 
        # To Naively discriminate in this wilderness type between 2 and 1 is chalenging 
        dataInt.loc[(dataInt[12] == 1) & (dataInt[0] > 3470),'PredictedCoverage'] = 7
        dataInt.loc[(dataInt[12] == 1) & (dataInt[0] > 3425)  & (dataInt[0] <= 3470),'PredictedCoverage'] = 1
        dataInt.loc[(dataInt[12] == 1) & (dataInt[0] <= 3425),'PredictedCoverage'] = 2
        dataInt.loc[(dataInt[12] == 1) & (dataInt[0] <= 2313),'PredictedCoverage'] = 3

        # Prediction for 4 wilderness type; It usually is 3
        dataInt.loc[(dataInt[13] == 1),'PredictedCoverage'] = 3
        return np.array(dataInt['PredictedCoverage'])  
    def predict_proba(self, X, y=None):
        tempX = self.predict(X)
        return np.vstack((tempX/7,1-tempX/7)).T
## NOTE!!! Such  classifier will have very poor performance!!! 