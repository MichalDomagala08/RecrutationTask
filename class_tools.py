
class ownScaler():
  def __init__(self,**kwargs):
    from sklearn.preprocessing import StandardScaler

    self.kwargs = kwargs
    self.sc = StandardScaler()
  def fit(self,X,y=None):
    X_temp = X.copy()

    column_nums= []
    for i in range(X_temp.shape[1]):
      if self.is_numeric(X[:,i]):
        column_nums.append(i)
    from numpy import hstack

  
    self.sc.fit(X_temp[:,column_nums])
  def is_numeric(self,X,y=None):
    from numpy import unique

    if len(unique(X)) > 7:
      return True
    else:
      return False

  def transform(self,X,y=None):
    X_temp = X.copy()
    from numpy import hstack
    column_nums = []
    n_col_numns = []
    for i in range(X_temp.shape[1]):
      if self.is_numeric(X_temp[:,i]):
        column_nums.append(i)
      else:
        n_col_numns.append(i)
    self.sc.fit(X_temp[:,column_nums])    
    X_temp[:,column_nums] = self.sc.transform(X_temp[:,column_nums])
    return hstack((X_temp[:,column_nums],X_temp[:,n_col_numns]))

  def fit_transform(self,X,y=None):
    from numpy import hstack
    X_temp = X.copy()
    column_nums= []
    n_col_numns = []
    for i in range(X_temp.shape[1]):
      if self.is_numeric(X_temp[:,i]):
        column_nums.append(i)
      else:
        n_col_numns.append(i)
    X_temp[:,column_nums] = self.sc.fit_transform(X_temp[:,column_nums])
    return hstack((X_temp[:,column_nums],X_temp[:,n_col_numns]))


def save_coefficients(classifier, filename):
    from h5py import File
    """Save the coefficients of a linear model into a .h5 file."""
    with File(filename, 'w') as hf:
        hf.create_dataset("coef",  data=classifier.coef_)
        hf.create_dataset("intercept",  data=classifier.intercept_)
        hf.create_dataset("classes", data=classifier.classes_)

def load_coefficients(classifier, filename):
    """Attach the saved coefficients to a linear model."""
    from h5py import File

    with File(filename, 'r') as hf:
        coef = hf['coef'][:]
        intercept = hf['intercept'][:]
        classes = hf['classes'][:]
    classifier.coef_ = coef
    classifier.intercept_ = intercept
    classifier.classes_ = classes
    return classifier

def step_decaygrid(epoch):
    import numpy as np
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop=3.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay(epoch):
    import numpy as np
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 30
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

def LossAccPlot(history):
    """
        Plots general history of model
    """
    import matplotlib.pyplot as plt
    
    fig,axes = plt.subplots(1,3)
    fig.set_figwidth(15)
    fig.set_figheight(5)

    axes[1].set_title('Accuracy: ')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    axes[1].plot(history.history["Accuracy"],label=['train'])
    axes[1].plot(history.history["val_Accuracy"],label=['test'])
    axes[1].legend()

    axes[0].set_title('Loss: ')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[0].plot(history.history["loss"],label=['train'])
    axes[0].plot(history.history["val_loss"],label=['test'])
    axes[0].legend()

    axes[2].set_title('ROC: ')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ROC')

    axes[2].plot(history.history["ROC"],label=['train'])
    axes[2].plot(history.history["val_ROC"],label=['test'])
    axes[2].legend()
    plt.savefig('.\\Results\\BestDNNHistory.png')