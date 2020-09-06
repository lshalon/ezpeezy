import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from .hyperparameter import HyperparameterSettings
from .data import DataManager
from .visualization import Visualizer

from tensorforce.environments import Environment

import random
import numpy as np
import math
import pandas as pd

class CustomEnvironment(Environment):
	"""
	This class represents the environment of your hyperparameter space. It handles
	new actions, internal states, logging, and fascilitates visualization.
	...
	Attributes
	----------
	_hps : HyperparameterSetting
		used to represent and manage your hyperparameter space
	_opt : string
		used to represent wether to maximize or minimize the monitored metric
	_model_type : string
		either 'sklearn" to indicaate that the given model is made with the sklearn 
		library or "keras" to indicate that the given model is made with the keras
		library
	_monitor_metric : string or functionn
		defines which metric from your model to monitor if model_type == 'keras' else
		defines the function of the metric to test on models' validation data
	_prev_reward : float
		the reward of the agent at the last training step
	_starting_tol : int/float
		the value to use as a episode ender if the metric does not increase by at each
		training step, or else end the agent's episode
	_tol_decay : int/float
		at each training step in the episode, decrease the tolerance by this value
	_curr_train_step : int
		the current training step of the agent in the current episode
	_curr_episode : int
		the current episode of the agent
	_history : pd.DataFrame
		the history of the agent and its tried parameters
	_model_train_batch_size : int
		the batch size to use when training your model
	_model_train_epochs : int
		number of eopchs to train your model for on each iteration
	build_model : function
		function that returns the model you want to optimize
	_data_manager : DataManager
		manages the data, cross validation to use for training you model
	_max_num_episodes : int
		represents the maximum number of episodes to be run for a given trial
	
	Methods
	-------
	states()
		Returns the configuration of the expected states.
	actions()
		Returns the configuration of the expected actions.
	max_episode_timesteps()
		Returns the maximum timesteps to take for a single episode before completion.
	set_k_folds(n_folds, pick_random)
		Specifies to the data manager what sort of cross-validation data 
		configuration to use.
	train_on_data(X_train, y_train, X_valid, y_valid)
		Specifies to the data manager what data to use for training.
	reset()
		Returns a random state and resets the environment.
	_get_cached_results(parameters)
		Returns false if the given parameters are already tried. Otherwise, it will
		return the result from when it was last tried.
	execute(actions)
		Returns next_state, terminal, reward after the provided actions are taken.
	get_history()
		Returns the history of the trial.
	reset_history()
		Resets the history of the trial.
	set_num_episodes(num_episodes)
		Sets the internal value for the maximum # of episodes to be run.
	get_best_params()
		Returns the best parameters in history for the specified optimization target.  
	"""
	
	def __init__(self, config, starting_tol, tol_decay, model_fn, model_type, 
					monitor_metric, opt, model_train_batch_size, model_train_epoch):
		"""
		Parameters
		----------
		config : dict
			a dictionary representing the configuration of the hyperparameter space.
			keys represent the name of the hyperparameter while keys can represent
			ranges of the parameter space and its type
		starting_tol : int/float
			the value to use as a episode ender if the metric does not increase by at each
			training step, or else end the agent's episode
		tol_decay : int/float
			at each training step in the episode, decrease the tolerance by this value
		model_fn : function
			function that returns the model you want to optimize
		model_type: string
			"sklearn" to signify that the passed in model_fn is of the sklearn library,
			or "keras" to signify that the passed in model_fn is made from the keras library
		monitor_metric : string or function
      the metric you would like to optimize in your model - string in the case of
      model_type == 'keras', function if model_type == 'sklearn' or None if to use
      the .score(X, y) function of the sklearn clasasifier
		opt : string
			used to represent wether to maximize or minimize the monitored metric
		model_train_batch_size : int
			the batch size to use when training your model
		model_train_epochs : int
			number of eopchs to train your model for on each iteration
		"""
		super().__init__()
		assert (opt == 'min') | (opt == 'max'), "opt param must be \'min' or \'max'"
		assert isinstance(starting_tol, float), "parameter \"{}\" must be a float".format(starting_tol)
		assert isinstance(tol_decay, float), "parameter \"{}\" must be a float".format(tol_decay)
		assert isinstance(model_train_batch_size, int), "parameter \"{}\" must be an int".format(model_train_batch_size)
		assert isinstance(model_train_epoch, int), "parameter \"{}\" must be an int".format(model_train_epoch)
		assert isinstance(model_type, str) & ((model_type == 'keras') | (model_type == 'sklearn')), \
								"model_type must be a string"

		self._hps = HyperparameterSettings(config)
		self._opt = opt # add constraint on input of this
		self._monitor_metric = monitor_metric
		self._prev_reward = 0. # should be more dynamic
		self._model_type = model_type

		self._starting_tol = starting_tol
		self._tol_decay = tol_decay

		self.curr_train_step = 1
		self.curr_episode = -1
		self.history = pd.DataFrame(columns=['episode'] + self._hps.get_parameter_labels() + [self._monitor_metric])

		self._model_train_batch_size = model_train_batch_size
		self._model_train_epoch = model_train_epoch
		self.build_model = model_fn

		self._data_manager = DataManager()

	def states(self):
		"""
		Returns the configuration of the expected states.
		
		Returns
		-------
		dict
			dict representation of the states and its size to expect
		"""
		return dict(type='float', shape=(len(self._hps.get_parameter_labels()) + 1,))

	def actions(self):
		"""
		Returns the configuration of the expected actions.

		Returns
		-------
		dict
			dict representation of the actions and its size to expect
		"""
		return dict(type='float', shape=(self._hps.get_num_actions(),), min_value=-1.0, max_value=1.0)

	def max_episode_timesteps(self):
		"""
		Returns the maximum timesteps to take for a single episode before completion.

		Returns
		-------
		int
			static integer representing maximum timesteps per episode
		"""
		return 50

	def set_k_folds(self, n_folds, pick_random):
		"""
		Specifies to the data manager what sort of cross-validation data 
		configuration to use.

		Parameters
		----------
		n_folds : int
			the number of folds to divide your dataset into using k-fold 
			cross-validation
		pick_random : int/None
			if set to an int, randomly select pick_random of the n_folds to use
			for training your model
		"""
		self._data_manager.set_k_fold(n_folds, pick_random)

	def train_on_data(self, X_train, y_train, X_test, y_test):
		"""
		Specifies to the data manager what data to use for training.
		
		Parameters
		----------
		X_train : iterible
			data used to train your model
		y_train : iterable
			labels used to train your model
		X_valid : iterable/None
			data used to validate your model unless using k-fold CV
		y_valid : iterable/None
			labels used to validate your model unless using k-fold CV
		"""

		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def reset(self):
		"""
		Returns a random state and resets the environment.

		Returns
		-------
		np.array
			array that represents the randomly generated state of a reset environment
		"""
		# plotting
		if self.curr_episode >= 0:
			Visualizer.plot_history(self.history, self._max_num_episodes, self._monitor_metric)
		
		state = list(self._hps.get_random_parameters().values())
		state += [self._prev_reward]
		state = np.array(state)

		self._prev_state = state

		self.curr_episode += 1

		next_state, terminal, reward = self.execute(np.zeros(self._hps.get_num_actions()))

		self.curr_train_step = 0

		return next_state

	def _get_cached_results(self, parameters):
		"""
		Returns false if the given parameters are already tried. Otherwise, it will
		return the result from when it was last tried.

		Parameters
		----------
		parameters : dict
			hyperparameters to search for in the history of runs
		
		Returns
		-------
		float/None
			returns a float if the given parameters exist in history or None if not
		"""
		results = self.history
		for (key, value) in parameters.items():
			results = results.loc[results[key] == value]
			if len(results) == 0:
				return None

		print('Using cached result for {}'.format([(k, '{:0.2f}'.format(parameters[k])) for k in parameters.keys()]))
		return list(results[self._monitor_metric])[0]

	def execute(self, actions):
		"""
		Returns next_state, terminal, reward after the provided actions are taken.

		Parameters
		----------
		actions : list
			actions made by the agent
		
		Returns
		-------
		next_state : np.array
			next state calculated by the actions taken
		terminal : boolean
			if the episosde should be terminated
		reward : float
			the reward of the taken action
		"""
		assert 0 <= len(actions) <= self._hps.get_num_actions()

		param_configs = self._hps.get_parameter_configs()
		config_ranges = [config[2] - config[1] for config in param_configs]
		
		delta = np.array([actions[i] * config_ranges[i] for i in range(len(config_ranges))])
		
		next_state = self._prev_state[:-1] + delta
		parameters = self._hps.get_feature_dictionary(next_state)

		self._internal_model = self.build_model(parameters)
		
		each_metric = []
		cached_results = self._get_cached_results(parameters)

		if type(cached_results) != type(None):
			each_metric.append(cached_results)
		else:
			for X_train, y_train, X_valid, y_valid in self._data_manager.feed_forward_data(self.X_train, self.y_train, self.X_test, self.y_test):
			
				if self._model_type == 'keras':
					history = self._internal_model.fit(X_train, y_train,
							batch_size=self._model_train_batch_size,
							epochs=self._model_train_epoch,
							verbose=0,
							validation_data=(X_valid, y_valid))

					each_metric.append(min(history.history[self._monitor_metric]) if self._opt == 'min' else max(history.history[self._monitor_metric]))
				elif self._model_type == 'sklearn':
					
					fitted_model = self._internal_model.fit(X_train, y_train)

					if type(self._monitor_metric) == type(None):
						each_metric.append(fitted_model.score(X_valid, y_valid))
					else:
						each_metric.append(self._monitor_metric(y_valid, fitted_model.predict(X_valid)))

		average_metric = sum(each_metric) / len(each_metric)
		reward = average_metric if self._opt == 'max' else -average_metric
	
		self.history.loc[len(self.history)] = [self.curr_episode] + list(parameters.values()) + [average_metric]
		
		print()
		print('Model with {} achieves {} of {:.5f}'.format([(k, '{:0.2f}'.format(parameters[k])) for k in parameters.keys()], 
																												self._monitor_metric, average_metric))

		terminal = False
		next_state = np.array(list(parameters.values()))
		next_state = np.append(next_state, reward)

		tol = self._starting_tol * math.pow(self._tol_decay, self.curr_train_step)

		diff = reward - self._prev_reward
		diff = diff if self._opt == 'min' else -diff

		if (self.curr_train_step > 1) & (diff < tol):
			print()
			print('Terminating episode, metric did not beat tolerance of {:0.5f}'.format(tol))
			self._prev_reward = 0.5 * reward if self._opt == 'max' else 2 * reward
			terminal = True
		else:
			self._prev_reward = reward

		self.curr_train_step += 1
		self._prev_state = next_state

		return next_state, terminal, reward

	def get_history(self):
		"""
		Returns the history of the agent including the configurations it has already
		tested.

		Returns
		-------
		pd.Dataframe
			Dataframe representing each absolute time step with its episode, configuration
			and monitored metric
		"""
		return self.history

	def reset_history(self):
		"""
		Resets the history of the trial.
		"""
		self.history = self.history.iloc[0:0]

	def set_num_episodes(self, num_episodes):
		"""
		Sets the internal value for the maximum # of episodes to be run.
		
		Parameters
		----------
		num_episodes : int
			the number of episodes taken as the maximum number of episodes in this run
		"""
		self._max_num_episodes = num_episodes

	def get_best_params(self):
		"""
		Returns the best parameters in history for the specified optimization target.

		Returns
		-------
		list
			best parameters for given goal
		"""
		if self._opt == 'min':
			return self.history.loc[self.history[self._monitor_metric].idxmin()]
		return self.history.loc[self.history[self._monitor_metric].idxmax()] 