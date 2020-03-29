import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from .hyperparameter import HyperparameterSettings
from tensorforce.environments import Environment

import random

batch_size = 128
num_classes = 10
epochs = 12

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
  
  def __init__(self, config, input_model=None, opt='max'):
    super().__init__()
    self._hps = HyperparameterSettings(config)
    self._opt = opt # add constraint on input of this
    self._prev_reward = 1e5 if opt == 'max' else -1e5
    self._tol_decay = 0.8
    self.curr_train_step = 0

  def build_model(self, custom_parameters):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    max_pool_size = custom_parameters['max_pool_size']

    model.add(MaxPooling2D(pool_size=(max_pool_size, max_pool_size)))
    model.add(Dropout(custom_parameters['dropout_rate']))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model


  def states(self):
    '''state_dict = OrderedDict()
    for param in self._hps.get_parameter_configs():
      if param[0] == 'int':
        number_states = param[1] - param[0] + 1
        state_dict.add({type=param[0], shape=(1,), num_states=number_states, min_value=param[1], max_value=param[2]})
      else:
        state_dict.add({type=param[0], shape=(1,), min_value=param[1], max_value=param[2]})

    state_dict.add({type='float', shape=(1,)}) # monitored metric
        
    return state_dict'''

    return dict(type='float', shape=(len(self._hps.get_parameter_labels()) + 1,))


  def actions(self):
    # return dict(type='float', num_actions=int(self._hps.get_num_actions())) #, min_value=-1, max_value=1)
    return dict(type='float', shape=(2,), min_value=-1.0, max_value=1.0)

  # Optional, should only be defined if environment has a natural maximum
  # episode length
  def max_episode_timesteps(self):
    return 50

  # Optional
  def close(self):
    super().close()

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
    
    history = self._internal_model.fit(x_train[indexes_to_use_for_training], y_train[indexes_to_use_for_training],
          batch_size=512,
          epochs=75,
          verbose=0,
          validation_data=(x_test, y_test))
    
    reward = -history.history['val_loss'][-1]
    print('Reward: {:0.5f}'.format(reward))
    terminal = False
    next_state = np.array(list(parameters.values()))
    next_state = np.append(next_state, reward)
    print('Next state: {}'.format(next_state))

    tol = -0.001 * math.pow(self._tol_decay, self.curr_train_step)

    if  self._prev_reward - reward  > tol or reward < -0.5:
      print()
      print('Terminating episode, prev_reward: {}, curr_reward: {}, tolerance: {:0.5f}'.format(self._prev_reward, reward, tol))
      self._prev_reward = 1e5 if self._opt == 'max' else -1e5
      terminal = True
    else:
      self._prev_reward = reward

    self.curr_train_step += 1
    self._prev_state = next_state
    return next_state, terminal, reward