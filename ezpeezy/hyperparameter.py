import re
import math
import random
from collections import OrderedDict 

class HyperparameterSettings():
	"""
	This class handles the representation of the hyperparameter space.
	...
	Attributes
	----------
	parameter_labels : list
		the names of the hyperparameters to use
	parameter_configs : list or tuples
		defines the type and bounds of a hyperparameter
	num_params : int
		number of parameters in hyperparameter space
	
	Methods
	-------
	_parameter_type(param_val)
		Returns a tuple representing the parameter's type, low and high bound from
		a given parameter configuration's value.
	_get_value(proposed_value, parameter_tuple)
		Returns the value of a proposed value that is in the hyperparameter space.
	get_parameter_configs()
		Returns the internal representation of the hyperparameter space.
	get_parameter_labels()
		Returns the name of each hyperparameter.
	get_feature_dictionary(proposed_value)
		Returns a dictionary whos keys are each the parameter label and the values are
		values in the parameter space that coorespond to the given proposed values.
	get_num_actions()
		Returns the number of actions that can be made on the hyperparameter space.
	get_random_parameters()
		Returns a random hyperparameter configuration.
	"""

	def __init__(self, hyp_config):
		"""
		Parameters
		----------
		hyp_config : dict
			represents the your hyperparameter space. The keys should be the names of the
			hyperparameters and the values should be defined as '0/1' for boolean, 'l:h'
			for float with bound l and h, and 'il:h' for integer with bound l and h
		"""

		self.parameter_labels = list(hyp_config.keys())
		self.parameter_configs = list()
		self.num_params = len(self.parameter_labels)

		for key in self.parameter_labels:
			self.parameter_configs.append(self._parameter_type(hyp_config[key]))

	def _parameter_type(self, param_val):
		"""
		Returns a tuple representing the parameter's type, low and high bound from
		a given parameter configuration's value.

		Parameters
		----------
		param_val : string
			one of the values of the given hyperparameter configuration
		
		Returns
		-------
		tuple
			tuple representing the (type, low_bound, high_bound) of the given hyperparameter
		"""
		assert isinstance(param_val, str), "parameter value \"{}\" must be a string".format(param_val)
		assert len(param_val) > 0, "parameter value \"{}\" must be of size > 0".format(param_val)

		if param_val == "0/1":
			return ("bool", 0, 1)
    
		integer = False
		if param_val[0] == 'i':
			integer = True
		param_val = param_val[1:]

		assert re.search('[a-zA-Z]', param_val) == None, "parameter value \"{}\" is an invalid value".format(param_val)

		low_value = param_val[:param_val.index(':')]
		ow_value = -1e4 if low_value == '' else float(low_value)
    
		high_value = param_val[param_val.index(':') + 1:]
		high_value = 1e4 if high_value == '' else float(high_value)
	
		assert low_value <= high_value, "parameter value \"{}\" is an invalid value".format(param_val)

		return ("int" if integer else "float", low_value, high_value)

	def _get_value(self, proposed_value, parameter_tuple):
		"""
		Returns the value of a proposed value that is in the hyperparameter space.

		Parameters
		----------
		proposed_value : float/int
			the proposed value for a given hyperparameter
		parameter_tuple : tuple
			representation of the hyperparameter space for a specific hyperparameter
		
		Returns
		-------
		float/int
			value that is in the hyperparameter space for the specified hyperparameter
		"""
		if proposed_value <= parameter_tuple[1]:
			return parameter_tuple[1] if parameter_tuple[0] == 'float' else int(round(parameter_tuple[1])) 
		if proposed_value >= parameter_tuple[2]:
			return parameter_tuple[2] if parameter_tuple[0] == 'float' else int(round(parameter_tuple[2]))
		return proposed_value if parameter_tuple[0] == 'float' else int(round(proposed_value))

	def get_parameter_configs(self):
		"""
		Returns the internal representation of the hyperparameter space.

		Returns
		-------
		list of tuples
			representation of the hyperparameter space
		"""
		return self.parameter_configs

	def get_parameter_labels(self):
		"""
		Returns the name of each hyperparameter.

		Returns
		-------
		list
			names of each hyperparameter in the hyperparameter space
		"""
		return self.parameter_labels

	def get_feature_dictionary(self, proposed_values):
		"""
		Returns a dictionary whos keys are each the parameter label and the values are
		values in the parameter space that coorespond to the given proposed values.

		Parameters
		----------
		proposed_values : iterable
			the proposed values of the hyperparameter space
		
		Returns
		OrderedDict
			representation of the hyperparameters to their valid values derived from 
			the proposed values provided
		"""
		return OrderedDict(zip(self.parameter_labels, [self._get_value(proposed_values[i], self.parameter_configs[i]) for i in range(self.num_params)]))

	def get_num_actions(self):
		"""
		Returns the number of actions that can be made on the hyperparameter space.

		Returns
		-------
		int
			the number of act ions an agent can take on this hyperparameter space
		"""
		return len(self.parameter_configs)

	def get_random_parameters(self):
		"""
		Returns a random hyperparameter configuration.

		Returns
		-------
		OrderedDict
			dictionary representing the hyperparameter space with random values
		"""
		random_state = [random.uniform(param[1], param[2]) for param in self.parameter_configs]
		return self.get_feature_dictionary(random_state)