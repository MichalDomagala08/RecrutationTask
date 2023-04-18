##### Loading Test Data 


#### Loading Libraries
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from class_tools import ownScaler

kern = "linear"
models = [] 

###############
### Reading Models
###############

#Read LogReg Model
from sklearn.linear_model import LogisticRegression
import pickle
LogReg = pickle.load(open('.\\Models\\LinearModel.sav', 'rb'))
models.append(('LogReg',LogReg))

#Read SVM model
if kern == "linear":
    import pickle
    SVM = pickle.load(open('.\\Models\\SVC.sav', 'rb'))

models.append(('SVM',SVM))

#Read DNN model
from tensorflow import keras
DNN = keras.models.load_model('.\\Models\\DNN')
models.append(('DNN',DNN))

# Import Naive Classifier - We cannot do Truthfully a Confusion Matrix!
import sys
sys.path.insert(1,'.\\Models')
from NaiveClassificator import NaiveCoverClassifer

naive = pickle.load(open('.\\Models\\Naive.sav', 'rb'))
models.append(('Naive',naive))

#Loading Test Data
X_test = np.load('.\\Data\\X_test.npy')
y_test = np.load('.\\Data\\y_test.npy')

# Colormaps for pretty plots
rainbow = cm.get_cmap('Spectral', 512).reversed()
newcmp = ListedColormap(rainbow(np.linspace(0.0, 0.9, 512)))

###############
### Confusion Matrices
###############

# making Confusion Matrices
for name, model in models:
    probs = model.predict(X_test)

    # if DNN we have to transform the output 
    if name != 'DNN':
        y_pred_temp = probs
    else: # When model is DNN we have to change OneHotEncoeded Data to our primary form
        y_pred_temp=np.array(pd.DataFrame(data=probs,columns=[1,2,3,4,5,6,7]).idxmax(1))

    # Plotting Confusion Matrix
    plt.title('Confusion Matrix: {}'.format(name))
    sns.heatmap(confusion_matrix(y_test,y_pred_temp),annot=True,cmap=newcmp,fmt='g')
    plt.savefig('.\\Results\\Confusion_'+name+'.png') # Saving Confusion matrices
    plt.clf()


accuracy_score = []
recall_score = []
f1_score = []
precision_score = []

###############
### Metrics
###############

# computing metrics: Accuracy Score, Precison, Recall, F1 
for name, model in models:
    print('Predicting: '+name)
    probs = model.predict(X_test)

    if (name != 'DNN') :
        y_pred_temp=probs
    else: # When model is DNN we have to change OneHotEncoeded Data to our primary form
        y_pred_temp=np.array(pd.DataFrame(data=probs,columns=[1,2,3,4,5,6,7]).idxmax(1))

    print()

    # Using Weighted Metrics
    accuracy_score.append(metrics.accuracy_score(y_test, y_pred_temp))
    precision_score.append(metrics.precision_score(y_test, y_pred_temp,average='weighted'))
    recall_score.append(metrics.recall_score(y_test, y_pred_temp,average='weighted'))
    f1_score.append( metrics.f1_score(y_test, y_pred_temp,average='weighted'))

d = {'precision_score': precision_score, 
    'recall_score': recall_score, 
    'f1_score': f1_score,
    'accuracy_score' : accuracy_score
    }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['LogReg','SVM','DNN','Naive'])
df.to_csv('.\\Results\\Results.csv') # Saving csv with classification metrics results

