import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from .hyperparameter import HyperparameterSettings
from .data import DataManager

from tensorforce.environments import Environment

import random
import numpy as np
import math

class CustomEnvironment(Environment):
  
  def __init__(self, config, starting_tol, tol_decay, input_model, opt_metric, opt):
    super().__init__()
    self._hps = HyperparameterSettings(config)
    self._opt = opt # add constraint on input of this
    self._opt_metric = opt_metric
    self._prev_reward = 1e5 if opt == 'max' else -1e5

    self._starting_tol = starting_tol
    self._tol_decay = tol_decay
    self.curr_train_step = 0
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
    state = list(self._hps.get_random_parameters().values())
    state += [self._prev_reward]
    state = np.array(state)
    self._prev_state = state
    self.curr_train_step = 0
    return state

  def execute(self, actions):
    print()
    print('Executing the following actions: {}'.format(actions))
    assert 0 <= len(actions) <= self._hps.get_num_actions()

    param_configs = self._hps.get_parameter_configs()
    config_ranges = [config[2] - config[1] for config in param_configs]
    
    delta = np.array([actions[i] * config_ranges[i] for i in range(len(config_ranges))])
   
    next_state = self._prev_state[:-1] + delta
    parameters = self._hps.get_feature_dictionary(next_state)

    print('Building model with {}'.format([(k, '{:0.2f}'.format(parameters[k])) for k in parameters.keys()]))
    self._internal_model = self.build_model(parameters)
    
    each_reward = []
    for X_train, y_train, X_valid, y_valid in self._data_manager.feed_forward_data(self.X_train, self.y_train, self.X_test, self.y_test):
      history = self._internal_model.fit(X_train, y_train,
            batch_size=512,
            epochs=75,
            verbose=1,
            validation_data=(X_valid, y_valid))
      
      each_reward.append(-min(history.history[self._opt_metric]) if self._opt == 'min' else max(history.history['val_loss']))
    
    reward = sum(each_reward) / len(each_reward)

    print('Reward: {:0.5f}'.format(reward))
    terminal = False
    next_state = np.array(list(parameters.values()))
    next_state = np.append(next_state, reward)
    print('Next state: {}'.format(next_state))

    tol = self._starting_tol * self._tol_decay * self.curr_train_step

    if reward - self._prev_reward < tol:
      print()
      print('Terminating episode, prev_reward: {}, curr_reward: {}, tolerance: {:0.5f}'.format(self._prev_reward, reward, tol))
      self._prev_reward = 1e5 if self._opt == 'max' else -1e5
      terminal = True
    else:
      self._prev_reward = reward

    self.curr_train_step += 1
    self._prev_state = next_state
    return next_state, terminal, reward