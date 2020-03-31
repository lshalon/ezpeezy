import re
import math
import random
from collections import OrderedDict 

class HyperparameterSettings():

  def __init__(self, parameters):
    self.parameter_labels = list(parameters.keys())
    self.parameter_configs = list()
    self.num_params = len(self.parameter_labels)

    for key in self.parameter_labels:
      self.parameter_configs.append(self._parameter_type(parameters[key]))

  def _parameter_type(self, parameter):
    assert isinstance(parameter, str), "parameter \"{}\" must be a string".format(parameter)
    assert len(parameter) > 0, "parameter \"{}\" must be of size > 0".format(parameter)

    if parameter == "0/1":
      return ("bool", 0, 1)
    
    integer = False
    if parameter[0] == 'i':
      integer = True
      parameter = parameter[1:]

    assert re.search('[a-zA-Z]', parameter) == None, "parameter \"{}\" is an invalid value".format(parameter)

    low_value = parameter[:parameter.index(':')]
    low_value = -1e4 if low_value == '' else float(low_value)
    
    high_value = parameter[parameter.index(':') + 1:]
    high_value = 1e4 if high_value == '' else float(high_value)

    assert low_value <= high_value, "parameter \"{}\" is an invalid value".format(parameter)

    return ("int" if integer else "float", low_value, high_value)

  def _get_value(self, value, parameter_tuple):
    if value <= parameter_tuple[1]:
      return parameter_tuple[1] if parameter_tuple[0] == 'float' else int(round(parameter_tuple[1])) 
    if value >= parameter_tuple[2]:
      return parameter_tuple[2] if parameter_tuple[0] == 'float' else int(round(parameter_tuple[2]))
    return value if parameter_tuple[0] == 'float' else int(round(value))

  def get_parameter_configs(self):
    return self.parameter_configs

  def get_parameter_labels(self):
    return self.parameter_labels

  def get_feature_dictionary(self, proposed_values):
    return OrderedDict(zip(self.parameter_labels, [self._get_value(proposed_values[i], self.parameter_configs[i]) for i in range(self.num_params)]))

  def get_num_actions(self):
    return len(self.parameter_configs)

  def get_random_parameters(self):
    random_state = [random.uniform(param[1], param[2]) for param in self.parameter_configs]
    return self.get_feature_dictionary(random_state)