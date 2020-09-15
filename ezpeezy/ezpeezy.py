from tensorforce.execution import Runner
from tensorforce.agents import DeepQNetwork
from .environment import CustomEnvironment

class Ezpeezy():
    """
    This class is used to encompass all user behavior and interactions.
    ...
    Attributes
    ----------
    _env : tensorforce.Environment
        custom environment used to define hyperparameter space and reward functions
    _agent : tensorforce.Agent
        dqn agent used to optimize the reward function defined in the environment
    _runner : tensorforce.Runner
        used to handle the training job of the agent
    
    Methods
    -------
    set_k_folds(n_folds, pick_random=None)
        Specifies to the environment what sort of cross-validation data 
        configuration to use.
    train_on_data(X_train, y_train, X_valid=None, y_valid=None)
        Specifies to the environment what data to use for training.
    get_history()
        Returns the history of the agent including the configurations it has already
        tested.
    run(num_episodes)
        Begins using the agent to discover the actions required to optimize the
        environment's reward.
    """

    def __init__(self, config, model_fn, model_type='sklearn', model_train_batch_size=256, 
                model_train_epochs=75, exploration=0.9, 
                exploration_decay_rate=0.8, monitor_metric='val_loss', 
                opt='max', starting_tol=-0.01, tol_decay=0.5):
        """
        Parameters
        ----------
        config : dict
            a dictionary representing the configuration of the hyperparameter space.
            keys represent the name of the hyperparameter while keys can represent
            ranges of the parameter space and its type
        model_fn : function
            function that returns the model you want to optimize
        model_type : string
            "sklearn" to signify that the passed in model_fn is of the sklearn library,
			or "keras" to signify that the passed in model_fn is made from the keras library
        model_train_batch_size : int
            the batch size to use when training your model
        model_train_epochs : int
            number of eopchs to train your model for on each iteration
        exploration : float
            the agent's exploration value
        exploration_decay_rate : float
            the agent's exploration value's decay rate (uses exponential decay)
        monitor_metric : None or string or function
            the metric you would like to optimize in your model - string in the case of
            model_type == 'keras', function if model_type == 'sklearn' or None if to use
            the .score(X, y) function of the sklearn clasasifier

            if function, defined to take in y_true, y_pred and return numeric type
        opt : string
            the optimization direction of the given monitor_metric
        starting_tol : int/float
            the value that you would like to see your metric to increase by at each
            training step, or else end the agent's episode
        tol_decay : int/float
            at each training step in the episode, decrease the tolerance by this value
        """

        self._env = CustomEnvironment(config, model_train_epoch=model_train_epochs,
                                    model_train_batch_size=model_train_batch_size, 
                                    model_fn=model_fn, model_type=model_type, 
                                    monitor_metric=monitor_metric, opt=opt, 
                                    starting_tol=starting_tol, tol_decay=tol_decay)
        self._agent = DeepQNetwork(states=self._env.states(), actions=self._env.actions(),
                     max_episode_timesteps=self._env.max_episode_timesteps(),
                     memory=60, batch_size=3,
                     exploration=dict(type='decaying', unit='timesteps', decay='exponential',
                                      initial_value=exploration, decay_steps=100000, decay_rate=exploration_decay_rate),
                     discount=dict(type='decaying', unit='timesteps', decay='exponential',
                                   initial_value=0.7, decay_steps=100000, decay_rate=0.5),
                     learning_rate=1e-20, l2_regularization=1e-3
                     )

        self.runner = Runner(agent=self._agent, environment=self._env)

    def set_k_folds(self, n_folds, pick_random=None):
        """
        Specifies to the environment what sort of cross-validation data 
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
        assert isinstance(n_folds, int), 'n_folds must be an int'
        assert (isinstance(pick_random, int) & (pick_random < n_folds)) | (pick_random == None) , \
             "pick  random must be an int less than n_folds or None" 

        self._env.set_k_folds(n_folds, pick_random)

    def train_on_data(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Specifies to the environment what data to use for training.

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
        self._env.train_on_data(X_train, y_train, X_valid, y_valid)

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
        return self._env.get_history()

    def run(self, num_episodes):
        """
        Begins using the agent to discover the actions required to optimize the
        environment's reward.
        
        Parameters
        ----------
        num_episodes : int
            number of episodes to try your agent for on your environment
        
        Prints
        ------
        the best parameters for your goal.
        """
        self._env.reset_history()
        self._env.set_num_episodes(num_episodes)
        self.runner.run(num_episodes=num_episodes)
        print('Best parameters are:')
        print(self._env.get_best_params())
        self.runner.close()