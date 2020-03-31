from sklearn.model_selection import KFold
import numpy as np
import random

class DataManager():
  
  def __init__(self):
    self.n_folds = None
    self._fold_seed = None
  
  def set_k_fold(self, n_folds, pick_random):
    self.n_folds = n_folds
    self.fold_selections = random.sample(range(n_folds), pick_random if pick_random != None else n_folds)

  def using_k_fold(self):
    return self.n_folds != None
  
  def feed_forward_data(self, X_train, y_train, X_valid=None, y_valid=None):
    assert ((type(X_valid) != type(None)) & (type(y_valid) != type(None))) | self.using_k_fold(), "use your agent's set_k_fold() method to set k_fold or provide validation data"
    if self._fold_seed == None:
        self._fold_seed = int(np.random.random() * 1e3)

    if self.using_k_fold():
        if len(self.fold_selections) == self.n_folds:
            print('Using {} folds for training and validation'.format(self.n_folds))
        else:
            print('Using {} of {} folds for training and validation'.format(len(self.fold_selections), self.n_folds))
        
        fold_counter = 0
        kfold = KFold(self.n_folds, shuffle=True, random_state=self._fold_seed)

        for fold in kfold.split(X_train, y_train):
            if fold_counter in self.fold_selections:
                yield X_train[fold[0]], y_train[fold[0]], X_train[fold[1]], y_train[fold[1]]
            fold_counter += 1
    else:
        yield X_train, y_train, X_valid, y_valid