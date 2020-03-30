from tensorforce.execution import Runner
from tensorforce.agents import DeepQNetwork
from .environment import CustomEnvironment

class Ezpeezy():
    def __init__(self, config, model_fn, opt_metric='val_loss', opt='max', starting_tol=0.01, tol_decay=0.5):
        self._env = CustomEnvironment(config, input_model=model_fn, opt_metric=opt_metric, opt=opt, 
                                starting_tol=starting_tol, tol_decay=tol_decay)
        self._agent = DeepQNetwork(states=self._env.states(), actions=self._env.actions(),
                     max_episode_timesteps=self._env.max_episode_timesteps(),
                     memory=60, batch_size=1, exploration=dict(type='decaying', unit='timesteps', decay='exponential',
                                                            initial_value=0.9, decay_steps=1000, decay_rate=0.8)
                     )

        self.runner = Runner(agent=self._agent, environment=self._env)

    def set_k_folds(self, n_folds, pick_random=None):
        self._env.set_k_folds(n_folds, pick_random)

    def train_on_data(self, X_train, y_train, X_valid=None, y_valid=None):
        self._env.train_on_data(X_train, y_train, X_valid, y_valid)

    def run(self, num_episodes):
        self.runner.run(num_episodes=num_episodes)
        self.runner.close()