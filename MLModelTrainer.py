
import pandas as pd 
import numpy as np
import ClasTrain
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

######################
### Reading Data:  ###
######################

##Firstly, determine column names from Info file:
# 10 explicitly named quantitatibve columns
# 4 binary columns regarding wilderness Area Designation
# 40 binary columns corresponding to soil type
# Cover_Type - our target

# We will disregard the explicit names of wilderness areas and soil types
 
soilTypes= ["Soil_Type_"+str(i) for i in range(40)]
wildernessType = ["wild_Des_"+str(i) for i in range(4)]
columns = ["Elevation", "Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"] + wildernessType + soilTypes +["Cover_Type"]
data = pd.read_csv('.\\Data\\covtype.data', sep=",",names = columns)

######################
### Preprocessing: ###
######################

from sklearn.model_selection import train_test_split
X = np.array(data)[:,0:53]
y = np.array(data)[:,54]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
np.save(".\\Data\\y_test.npy",y_test)
np.save(".\\Data\\X_test.npy",X_test)
kfold = StratifiedKFold(5)

####################
### Classifiers: ###
####################

###### LogReg Classifier 
"""
model_linear = ClasTrain.LogReg_Classifer(X_train,y_train,kfold,gridSearch=True)
###### Save Models
print("Saving model...")
pickle.dump(model_linear,open('.\\Models\\LinearModel.sav','wb'))

from sklearn.metrics import accuracy_score
preds = model_linear.predict(X_test)
print(accuracy_score(preds,y_test))
"""
###### SVC Classifier 
"""
svcClassifier = ClasTrain.SVC_Classifer(X_train,y_train,kfold,kern="linear",gridSearch=False)
print("Saving model...")
filename = 'C:\\Users\\barak\\Downloads\\RecrutationTask\\Models\\SVC.sav'
pickle.dump(svcClassifier, open(filename, 'wb'))"""

####### DNN Model
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
    


# Learning Rate Scheduler
from class_tools import step_decaygrid
#lrate = LearningRateScheduler(step_decaygrid)

#History
history = History()

#Early Stopping
early_stopping = EarlyStopping(monitor='val_ROC', patience=100, mode='max', verbose=0,restore_best_weights=True)

#Searching for best parameters:
# Searched: First FC Layer size, Second FC Layer size, Regularization
from ClasTrain import build_model,gridNN
gridSearchDnn = True
y_train = to_categorical(y_train)

if gridSearchDnn:
  keras_class = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)
  rnd_search_cv =gridNN(keras_class)
 
  rnd_search_cv.fit(X_train, y_train[:,1:], validation_split =0.2,
                    class_weight =  { 0:class_weights[0],
                                      1:class_weights[1],
                                      2:class_weights[2],
                                      3:class_weights[3],
                                      4:class_weights[4], 
                                      5:class_weights[5],
                                      6:class_weights[6]}, 
                    batch_size=256,epochs=5,
                    callbacks=[history,early_stopping])

  best_estim = {'lr':rnd_search_cv.best_params_['lr'],
                'drp':rnd_search_cv.best_params_['drp'],
                'optim':rnd_search_cv.best_params_['optim']}
else:
  best_estim = {'lr' : 0.1,
                'drp' : 0,
                'optim' : "adam"}

### Fit model based on searched parameters
from keras.metrics import AUC
pickle.dump(best_estim, open('.\\Results\\gridDNN.sav', 'wb'))
y_test_cat = to_categorical(y_test)
history = History()
from class_tools import step_decay
lrate = LearningRateScheduler(step_decay)

DNN = build_model(
                  optim=best_estim['optim'],
                  lr=best_estim['lr'],
                  drp=best_estim['drp'],
                  metrics=[AUC(from_logits=True,name='ROC'),"Accuracy"])
DNN.fit(X_train, y_train[:,1:], validation_split=0.2,
                  class_weight =  { 0:class_weights[0],
                                    1:class_weights[1],
                                    2:class_weights[2],
                                    3:class_weights[3],
                                    4:class_weights[4], 
                                    5:class_weights[5],
                                    6:class_weights[6]
                                  }, 
                  batch_size=256,epochs=150,
                  callbacks=[history,lrate,early_stopping])

DNN.save('.\\Models\\DNN')

from sklearn.metrics import accuracy_score
print(DNN.evaluate(X_test,y_test_cat[:,1:],batch_size=128))
print(DNN.predict(X_test))
print(accuracy_score(np.array(pd.DataFrame(data=DNN.predict(X_test),columns=[1,2,3,4,5,6,7]).idxmax(1)),y_test))
from class_tools import LossAccPlot
LossAccPlot(history)





