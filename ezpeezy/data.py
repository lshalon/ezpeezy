from sklearn.model_selection import KFold

class DataManager():
  
  def __init__(self):
    self.n_folds = None
  
  def set_k_fold(self, n_folds, pick_random):
    self.n_folds = n_folds
    self.fold_selections = random.sample(np.arange(n_folds), pick_random if pick_random != None else n_folds)

  def using_k_fold(self):
    return self.n_folds != None
  
  def feed_forward_data(self, X_train, Y_train, x_valid=None, y_valid=None):
    assert (x_valid != None & y_valid != None) | self.use_k_folds, "use your agent's set_k_fold() method to set k_fold or provide validation data"
    if self._fold_seed == None:
        self._fold_seed = int(np.random() * 1e3)

    if self.use_k_fold():
        if len(self.fold_selections) == self.n_folds:
            print('Using {} folds for training and validation'.format(self.n_folds))
        else:
            print('Using {} of {} folds for training and validation'.format(len(self.fold_selections), self.n_folds))
        fold_counter = 0
        kfold = KFold(n_folds, random_state=self._fold_seed)
        for X_t, x_v, Y_t, y_v in (kfold.split(X_train), kfold.split(Y_train)):
            if fold_counter in self.fold_selections:
                yield X_t, Y_t, x_v, y_v
            fold_counter += 1
    else:
        yield X_train, Y_train, x_valid, y_valid