#### Loading and Exploratory Analysis
## 

## imports
# %%
import pandas as pd 
import sklearn as sc 
import seaborn as sb
import numpy as np
##Firstly, determine cvolumn names from Info file:
# 10 explicitly named quantitatibve columns
# 4 binary columns regarding wilderness Area Designation
# 40 binary columns corresponding to soil type
# Cover_Type - our target

# We will disregard the explicit names of wilderness areas and soil types
 # %% 
soilTypes= ["Soil_Type_"+str(i) for i in range(40)]
wildernessType = ["wild_Des_"+str(i) for i in range(4)]
columns = ["Elevation", "Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"] + wildernessType + soilTypes +["Cover_Type"]
data = pd.read_csv('.\\Data\\covtype.data', sep=",",names = columns)
# %%
import sys
sys.path.insert(1,'.\\Models')

from NaiveClassificator import NaiveCoverClassifer

cover = NaiveCoverClassifer()
import pickle

filename = 'C:\\Users\\barak\\Downloads\\RecrutationTask\\Models\\Naive.sav'
pickle.dump(cover, open(filename, 'wb'))
# %% 
#zeroing autocorelations for convienience
correlationData =np.where(np.array(data.corr()['Cover_Type']) !=1 ,np.array(data.corr()['Cover_Type']),0)

#Determine which 
correlationData[abs(correlationData) >0.3]
np.where(abs(correlationData)>0.3)[0][0]

columns[np.where(abs(correlationData)>0.3)[0][0]]

# %%
correlationData
# as we can see the data are already OneHotEncoded:

#Naive Classifier will regard only those Variables with greatest correlation with CoverType
# Naive Classifier is based on Wilderness Types as specified in DataInfo:
# Neota ( Wild2) is usually of Type one
# Rawah and Comanche Peak (Wild1, Wild3) is of type 2,1 and then type 5
# Cache La Pudre ( wild4) consist of type 3 6 and 4 
# %% 
#Table Regarding

# Creating Easier to read table without one-hot encoded variables
readingData = data.copy()
readingData.drop(['wild_Des_0','wild_Des_1','wild_Des_2','wild_Des_3'],axis=1)
readingData.drop(soilTypes,axis=1,inplace=True)
readingData.drop('Cover_Type',axis=1,inplace=True)
readingData['Wilderness'] =  pd.get_dummies(data[['wild_Des_0','wild_Des_1','wild_Des_2','wild_Des_3']]).idxmax(1)
readingData['Soil_Type'] = pd.get_dummies(data[soilTypes]).idxmax(1)
readingData['Cover_Type'] = data['Cover_Type']


WildernessTable = pd.pivot_table(data,columns=['Cover_Type'],index = ['Wilderness_Area']
                        aggfunc='count')['Aspect']
WildernessTable
# %%
readingData


# %%

SlopeTable = pd.pivot_table(readingData,index=['Wilderness'],columns=['Cover_Type'],values=['Elevation'],
                        aggfunc=np.max)
SlopeTable
# %%

# %% Naive Classifier

# Requiers Data in DataFrame or Numpy array 
# with crucially with the same structure as loaded data

dataInt = pd.DataFrame(np.array(data))
dataInt['PredictedCoverage'] = np.zeros(len(dataInt))
#  PRedicitons for 2 wilderness types
dataInt.loc[(dataInt[11] == 1),'PredictedCoverage'] = 1
 
# Predictions for 1 wilderness type 
# To Naively discriminate in this wilderness type between 2 and 1 is chalenging 
dataInt.loc[(dataInt[10] == 1) & (dataInt[0] > 3333),'PredictedCoverage'] = 1
dataInt.loc[(dataInt[10] == 1) & (dataInt[0] <= 3333),'PredictedCoverage'] = 2
dataInt
# Predictions for 3 wilderness type 
# To Naively discriminate in this wilderness type between 2 and 1 is chalenging 
dataInt.loc[(dataInt[12] == 1) & (dataInt[0] > 3470),'PredictedCoverage'] = 7
dataInt.loc[(dataInt[12] == 1) & (dataInt[0] > 3425),'PredictedCoverage'] = 1
dataInt.loc[(dataInt[12] == 1) & (dataInt[0] <= 3425),'PredictedCoverage'] = 2
dataInt.loc[(dataInt[12] == 1) & (dataInt[0] <= 2313),'PredictedCoverage'] = 3
  
# Prediction for 4 wilderness type; It usually is 3
dataInt.loc[(dataInt[13] == 1),'PredictedCoverage'] = 3
dataInt[[0,10,11,12,13,54,'PredictedCoverage']]  
## NOTE!!! Such  classifier will have very poor performance!!! 
# %%
dataInt[(dataInt[11] == 1)]

# %%

### Preprocessing:

### Standard Scaling 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

X = np.array(data)[:,0:53]
y = np.array(data)[:,54]
"""
        class OwnScale(StandardScaler):

            def __init__(self):
                self.sc = StandardScaler()
                pass

            def fit(self,X,y=None):
                self.X_scalable = X[:,0:9]
                self.X_unscalable = X[:,10:]
                self.sc.fit(self.X_scalable)
                
            def transform(self,X,y=None):
                X_scalable = X[:,0:9]
                X_unscalable = X[:,10:]
                X_Scaled = sc.transform(X_scalable)

                return np.hstack((X_Scaled,X_unscalable))
"""

###### Preprocessing Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Splitting and Standard Scaling 
sc = StandardScaler()
sc.fit(X_train[:,0:9])


X_train_sc = sc.transform(X_train[:,0:9])
pca = PCA(n_components=5)
# X_train_sc = pca.transform(X_train_sc)
X_train = np.hstack((X_train_sc,X_train[:,10:13]))
kfold = StratifiedKFold(5)
"""# %%
### Logistic Regression
param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
}

grid_2 = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, return_train_score=True)
grid_2.fit(X_train, y_train)
grid_2.best_estimator_

model1 = LogisticRegression(C=grid_2.best_estimator_.C
).fit(X_train, y_train)

preds1 = model1.predict(np.hstack(((sc.transform(X_test[:,0:9])),X_test[:,10:13])))

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
print('f1_score: {}'.format(f1_score(y_test,preds1,average=('weighted'))))
print('accuracy_score: {}'.format(accuracy_score(y_test,preds1)))
print('precsion_score: {}'.format(precision_score(y_test,preds1,average=('weighted'))))
print('recall_score: {}'.format(recall_score(y_test,preds1,average=('weighted'))))# %%



###### SVC Classifier 
svcClassifier = SVC(kernel = 'rbf')
param_grid = {
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
}

grid_1 = GridSearchCV(svcClassifier, param_grid, cv=kfold, return_train_score=True)
#grid_1.fit(X_train, y_train)
#grid_1.best_estimator_

model2 = svcClassifier(C=grid_1.best_estimator_.C,gamma=grid_1.best_estimator_.gamma).fit(X_train, y_train)
preds2 = model2.predict(np.hstack((sc.transform(X_test[:,0:9]),X_test[:,10:])) )

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
print('f1_score: {}'.format(f1_score(y_test,preds2,average=('weighted'))))
print('accuracy_score: {}'.format(accuracy_score(y_test,preds2)))
print('precsion_score: {}'.format(precision_score(y_test,preds2,average=('weighted'))))
print('recall_score: {}'.format(recall_score(y_test,preds2,average=('weighted'))))
"""

import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
##########################################################
####### DL MODEL ##################

from keras.callbacks import History
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

#Computing class Weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes =np.unique(np.array(y_train)),
                                            y=np.array(y_train))
    
print(class_weights)



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Learning Rate Scheduler
from class_tools import step_decay
#lrate = LearningRateScheduler(step_decay)

#History
history = History()

#Early Stopping
early_stopping = EarlyStopping(monitor='val_ROC', patience=100, mode='max', verbose=0,restore_best_weights=True)

#Searching for best parameters
from ClasTrain import build_model,gridNN
keras_class = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)
rnd_search_cv =gridNN(keras_class)
rnd_search_cv.fit(X_train, y_train[:,1:], validation_data= (np.hstack((sc.transform(X_test[:,0:9]),X_test[:,10:13])),y_test[:,1:]),class_weight =  {0:class_weights[0],1:class_weights[1],2:class_weights[2],3 : class_weights[3],4 : class_weights[4], 5: class_weights[5],6 :class_weights[6]}, batch_size=256,epochs=5, callbacks=[history,early_stopping])
# %%
y_train = to_categorical(y_train)
y_train
# %%
y_train
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

np.unique(y_train)


# %%
from sklearn.metrics import get_scorer_names
get_scorer_names()
# %%
