


from keras.metrics import AUC
import tensorflow as tf

def LogReg_Classifer(X_train,y_train,kfold,gridSearch=True):
    """
        Function used to train Logistic Regression Classifier
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from class_tools import ownScaler

    ###### LogReg Classifier 
    print('######## Log Reg Classifier ##########')

    # gridSearch parameter is used to turn on and off Grid Searching od best parameters
    if gridSearch:
        print("Grid Search for best reularization...")
        param_grid = {
            'LogReg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        }

        # Grid Searching variables - Making pipeline with Custom scaler
        grid_2 = GridSearchCV(Pipeline([('scaler', ownScaler()), ('LogReg',LogisticRegression())]) , param_grid, cv=kfold, return_train_score=True,verbose=2)
        grid_2.fit(X_train, y_train)
        estim = {'C': grid_2.best_params_['LogReg__C'] }
    else:
        estim= {'C' : 1}

    print('Training best parameter model...')
    # training the final model

    model_Linear = Pipeline([('scaler', ownScaler()), ('LogReg', LogisticRegression(C=estim['C']))])
    model_Linear.fit(X_train,y_train)
    
    return model_Linear


def SVC_Classifer(X_train,y_train,kfold,gridSearch=False,kern="linear"):
    """
        Function used to train SVC_Classifer

        Currently the grid search parameter is False due to substantial time of computation
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from class_tools import ownScaler
    ###### SVC Classifier 
    
    print('######## SVCClassifier ##########')

    # Making pipeline with Custom scaler
    svcClassifier = Pipeline([('scaler', ownScaler()), ('SVC',SVC())])

    # gridSearch parameter is used to turn on and off Grid Searching od best parameters
    if gridSearch:
        print("     Grid Search for best reularization...")
        param_grid = {
            'SVC__gamma': [ 0.01, 0.1, 1],
            'SVC__C': [ 0.01, 0.1, 1],
        }

        grid_1 = GridSearchCV(svcClassifier, param_grid, cv=kfold, return_train_score=True,verbose=2)
        grid_1.fit(X_train, y_train)

        estim = {'C': grid_1.best_params_['SVC__C'],'gamma':grid_1.best_params_['SVC__gamma']}
        
    else:
        estim = {'C': 1,'gamma':0.1} # seved in values

    print('    Training best parameter model...')
    # training the final model
    model_SVC = Pipeline([('scaler', ownScaler()), ('SVC', SVC(C=estim['C'],gamma=estim['gamma'],kernel=kern,verbose=True))])
    model_SVC.fit(X_train, y_train)
    return model_SVC


def build_model(optim="adam",
                lr=0.01,drp=0.3,metrics=AUC(from_logits=True,name='ROC')):
    
    from keras.layers import Dense,BatchNormalization
    from keras.models import Sequential
    from keras.layers import Dense,Activation, Dropout
    from keras.regularizers import l1
    from keras.backend import set_value
    
    # Creating Model
    model_cat = Sequential()
    model_cat.add(Dense(1000,input_shape=(53,))) # Input shape is seved in for convienience
    model_cat.add(BatchNormalization())
    model_cat.add(Activation("relu"))
    model_cat.add(Dropout(drp)) # Dropout parameter is gridSearched!
    model_cat.add(Dense(500))
    model_cat.add(BatchNormalization())
    model_cat.add(Activation("relu"))
    model_cat.add(Dropout(drp))
    model_cat.add(Dense(7,activation="softmax"))
    model_cat.compile(loss="categorical_crossentropy",optimizer=optim, metrics=metrics)

    # Setting the Optimizer Learning rate Value!
    set_value(model_cat.optimizer.learning_rate,lr)
    return model_cat
    

def gridNN(keras_class):
    """
        Randomized Grid Search for DNN model. Needs model wrapped in kerasClassifier class
    """
    import tensorflow as tf

    from sklearn.model_selection import RandomizedSearchCV
    param_distribs = {
        "optim": ["adam","rmsprop"],
        "lr":[0.001,0.01,0.1],
        "drp":[0,0.3,0.8]
    }

    rnd_search_cv = RandomizedSearchCV(keras_class, param_distribs, cv=5, verbose=1,scoring='roc_auc')
    return rnd_search_cv


