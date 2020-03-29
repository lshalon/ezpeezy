from sklearn.model_selection import KFold

class DataManager():
  
  def __init__(self):
    self.n_folds = None
  
  def set_k_fold(self, n_folds, pick_random):
    self.n_folds = n_folds
    self.fold_selections = random.sample(np.arange(n_folds), pick_random if pick_random != None else n_folds)

  def use_k_fold(self):
    return n_folds != None
  
  def feed_forward_data(self, X_train, Y_train, x_valid=None, y_valid=None):
    assert (x_valid != None & y_valid != None) | self.use_k_folds, "use your agent's set_k_fold() method to set k_fold or provide validation data"

    if self.use_k_fold():
      fold_counter = 0
      kfold = KFold(n_folds)
      for X_t, x_v, Y_t, y_v in (kfold.split(X_train), kfold.split(Y_traain)):
        if folder_counter in self.fold_selections:
          yield X_t, Y_t, x_v, y_v
        fold_counter += 1
    else:
      yield X_train, Y_train, x_valid, y_valid