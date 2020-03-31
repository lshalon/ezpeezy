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
  
  def __init__(self, config, starting_tol, tol_decay, input_model, opt_metric, opt, 
              model_train_batch_size, model_train_epoch):
    super().__init__()
    self._hps = HyperparameterSettings(config)
    self._opt = opt # add constraint on input of this
    self._monitor_metric = opt_metric
    self._prev_reward = 1e5 if opt == 'max' else -1e5

    self._starting_tol = starting_tol
    self._tol_decay = tol_decay

    self.curr_train_step = 1
    self.curr_episode = -1
    self.history = pd.DataFrame(columns=['episode'] + self._hps.get_parameter_labels() + [self._monitor_metric])

    self._model_train_batch_size = model_train_batch_size
    self._model_train_epoch = model_train_epoch
    self.build_model = input_model

    self._data_manager = DataManager()

  def states(self):
    return dict(type='float', shape=(len(self._hps.get_parameter_labels()) + 1,))


  def actions(self):
    # return dict(type='float', num_actions=int(self._hps.get_num_actions())) #, min_value=-1, max_value=1)
    return dict(type='float', shape=(self._hps.get_num_actions(),), min_value=-1.0, max_value=1.0)

  # Optional, should only be defined if environment has a natural maximum
  # episode length
  def max_episode_timesteps(self):
    return 50

  # Optional
  def close(self):
    super().close()

  def set_k_folds(self, n_folds, pick_random):
    self._data_manager.set_k_fold(n_folds, pick_random)

  def train_on_data(self, X_train, y_train, X_test, y_test):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

  def reset(self):
    # plotting
    if self.curr_episode >= 0:
      Visualizer.plot_history(self.history, self._max_num_episodes, self._monitor_metric)
    
    state = list(self._hps.get_random_parameters().values())
    state += [self._prev_reward]
    state = np.array(state)
    self._prev_state = state
    self.curr_train_step = 0
    self.curr_episode += 1
    return state

  def execute(self, actions):
    assert 0 <= len(actions) <= self._hps.get_num_actions()

    param_configs = self._hps.get_parameter_configs()
    config_ranges = [config[2] - config[1] for config in param_configs]
    
    delta = np.array([actions[i] * config_ranges[i] for i in range(len(config_ranges))])
   
    next_state = self._prev_state[:-1] + delta
    parameters = self._hps.get_feature_dictionary(next_state)
    for val in parameters.values():
      print(type(val))

    self._internal_model = self.build_model(parameters)
    
    each_metric = []
    for X_train, y_train, X_valid, y_valid in self._data_manager.feed_forward_data(self.X_train, self.y_train, self.X_test, self.y_test):
      history = self._internal_model.fit(X_train, y_train,
            batch_size=self._model_train_batch_size,
            epochs=self._model_train_epoch,
            verbose=0,
            validation_data=(X_valid, y_valid))
      
      each_metric.append(min(history.history[self._monitor_metric]) if self._opt == 'min' else max(history.history[self._monitor_metric]))
    
    average_metric = sum(each_metric) / len(each_metric)
    reward = average_metric if self._opt == 'max' else -average_metric
  
    self.history.loc[len(self.history)] = [self.curr_episode] + list(parameters.values()) + [average_metric]

    print('Model with {} achieves {} of {:.5f}'.format([(k, '{:0.2f}'.format(parameters[k])) for k in parameters.keys()], 
                                                        self._monitor_metric, average_metric))

    terminal = False
    next_state = np.array(list(parameters.values()))
    next_state = np.append(next_state, reward)

    tol = self._starting_tol * self._tol_decay * self.curr_train_step

    if reward - self._prev_reward < tol:
      print()
      print('Terminating episode, metric did not beat tolerance of {:0.5f}'.format(tol))
      self._prev_reward = 1e5 if self._opt == 'max' else -1e5
      terminal = True
    else:
      self._prev_reward = reward

    self.curr_train_step += 1
    self._prev_state = next_state
    return next_state, terminal, reward

  def get_history(self):
    return self.history

  def reset_history(self):
    self.history = self.history.iloc[0:0]

  def set_num_episodes(self, num_episodes):
    self._max_num_episodes = num_episodes

  def get_best_params(self):
    if self._opt == 'min':
      return self.history.loc[history[self._monitor_metric].idxmin()]
    return self.history.loc[history[self._monitor_metric].idxmax()] 