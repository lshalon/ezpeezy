import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from .hyperparameter import HyperparameterSettings
from .data import DataManager
from tensorforce.environments import Environment

import random
import numpy as np
import math

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

indexes_to_use_for_training = random.sample(range(len(x_train)), int(len(x_train) / 2))

class CustomEnvironment(Environment):
  
  def __init__(self, config, starting_tol, tol_decay, input_model=None, opt='max'):
    super().__init__()
    self._hps = HyperparameterSettings(config)
    self._opt = opt # add constraint on input of this
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
            verbose=0,
            validation_data=(X_valid, y_valid))
      
      each_reward.append(-min(history.history['val_loss']) if self._opt == 'min' else max(history.history['val_loss']))
    
    reward = sum(each_reward) / len(each_reward)

    print('Reward: {:0.5f}'.format(reward))
    terminal = False
    next_state = np.array(list(parameters.values()))
    next_state = np.append(next_state, reward)
    print('Next state: {}'.format(next_state))

    tol = self._starting_tol * self._tol_decay * self.curr_train_step

    if reward - self._prev_reward < tol or reward < -0.5:
      print()
      print('Terminating episode, prev_reward: {}, curr_reward: {}, tolerance: {:0.5f}'.format(self._prev_reward, reward, tol))
      self._prev_reward = 1e5 if self._opt == 'max' else -1e5
      terminal = True
    else:
      self._prev_reward = reward

    self.curr_train_step += 1
    self._prev_state = next_state
    return next_state, terminal, reward