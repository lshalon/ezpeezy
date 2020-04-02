from sklearn.model_selection import KFold
import numpy as np
import random

class DataManager():
	"""
	This class represents a data manager for all sorts of data techniques required
	for the model to by optimized's training.
	...
	Attributes
	----------
	n_folds : int/None
		the number of folds used for k-fold cross-validation unless specified
		to none
	_pick_random : int/None
		the number of folds to use of the k-fold cross-validation technique unless
		specified to none
	fold_selection : list
		the folds to use for k-fold technique
	_fold_seed : int
		the seed to use for shuffling the data. This will stay static

	Methods
	-------
	set_k_fold(n_folds, pick_random) 
		Defines the k-fold cross-validation technique to use for training.
	using_k_fold()
		Returns true if the selected technique is k-fold cross-validation.
	feed_forward_data(X_train, y_train, X_valid, y_valid)
		Yields the correct data depending on the validation technique used.
	"""
	
	def __init__(self):
		"""
		Parameters
		----------
		"""

		self.n_folds = None
		self._fold_seed = None
	
	def set_k_fold(self, n_folds, pick_random):
		"""
		Defines the k-fold cross-validation technique to use for training.

		Parameters
		----------
		n_folds : int
			the number of folds to use for the k-fold cross-validation technique
		pick_random : int/None
			if an integer, pick randomly this many folds from the n_folds to use across
			validations. Effectively, this reduces the number of folds actually processed.
		"""
		self.n_folds = n_folds
		self.fold_selections = random.sample(range(n_folds), pick_random if pick_random != None else n_folds)

		print('Using {} of {} folds for training and validation'.format(len(self.fold_selections), self.n_folds))
				
	def using_k_fold(self):
		"""
		Returns true if the selected technique is k-fold cross-validation.

		Returns
		-------
		boolean
			true if the user has selected to use k-fold cross-validation, false otherwise
		"""
		return self.n_folds != None
	
	def feed_forward_data(self, X_train, y_train, X_valid=None, y_valid=None):
		"""
		Yields the correct data depending on the validation technique used.

		Parameters
		----------
		X_train : iterable
			data to use for training or for the k-fold CV if it is enabled
		y_train : iterable
			labels to use for training or for the k-fold CV if it is enabled
		X_valid : iterable/None
			data to use for validation. If k-fold CV is enabled, this is ignored
		y_valid : iterable/None
			labels to use for validation. If k-fold CV is enabled, this is ignored
		"""
		assert ((type(X_valid) != type(None)) & (type(y_valid) != type(None))) | self.using_k_fold(), "use your agent's set_k_fold() method to set k_fold or provide validation data"
		if self._fold_seed == None:
			self._fold_seed = int(np.random.random() * 1e3)

		if self.using_k_fold():
			fold_counter = 0
			kfold = KFold(self.n_folds, shuffle=True, random_state=self._fold_seed)

			for fold in kfold.split(X_train, y_train):
				if fold_counter in self.fold_selections:
					yield X_train[fold[0]], y_train[fold[0]], X_train[fold[1]], y_train[fold[1]]
				fold_counter += 1
		else:
			yield X_train, y_train, X_valid, y_valid