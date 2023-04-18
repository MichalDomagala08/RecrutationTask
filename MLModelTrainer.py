
import pandas as pd 
import numpy as np
import pickle
import ClasTrain
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

# getting the target and data from our DF
from sklearn.model_selection import train_test_split
X = np.array(data)[:,0:53]
y = np.array(data)[:,54]

# Train test Split and saving o test variables for further testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
np.save(".\\Data\\y_test.npy",y_test)
np.save(".\\Data\\X_test.npy",X_test)
kfold = StratifiedKFold(5)

####################
### Classifiers: ###
####################

###### LogReg Classifier 

#Learning via  own script best possible LogReg
model_linear = ClasTrain.LogReg_Classifer(X_train,y_train,kfold,gridSearch=True)
###### Save Models
print("Saving model...")
pickle.dump(model_linear,open('.\\Models\\LinearModel.sav','wb'))

###### SVC Classifier 

#Learning via own function  best possible SVM
svcClassifier = ClasTrain.SVC_Classifer(X_train,y_train,kfold,kern="linear",gridSearch=False)
print("Saving model...")
filename = 'C:\\Users\\barak\\Downloads\\RecrutationTask\\Models\\SVC.sav'
pickle.dump(svcClassifier, open(filename, 'wb'))

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
    

#History
history = History()

#Early Stopping - proved to be irrelevant, but in further iterations may be of hand
# Using maximal validational ROC as a stopping metric
early_stopping = EarlyStopping(monitor='val_ROC', patience=25, mode='max', verbose=0,restore_best_weights=True)

#Searching for best parameters:
# Searched: dropout rate, learning rate, oprimizer - Adam vs rmsprop
from ClasTrain import build_model,gridNN
gridSearchDnn = True

#OneHotEncoding target variable as it is necessery for learning DNN
y_train = to_categorical(y_train)

# gridSearchDNN similarily is responsible for turning of GridSearch 
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
else: # when the gridSearch is off
  best_estim = {'lr' : 0.1,
                'drp' : 0,
                'optim' : "adam"} # Seved in values proven by trial and error to be quite stable

### Fit model based on searched parameters

from keras.metrics import AUC
pickle.dump(best_estim, open('.\\Results\\gridDNN.sav', 'wb')) # Saving whole GridSearch for further analysis
y_test_cat = to_categorical(y_test)
history = History()

# Importing and using step decay as Learning Rate Scheduler
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
 
# Saving Model
DNN.save('.\\Models\\DNN')

from class_tools import LossAccPlot
LossAccPlot(history)





