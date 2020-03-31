from tensorforce.execution import Runner
from tensorforce.agents import DeepQNetwork
from .environment import CustomEnvironment

class Ezpeezy():
    def __init__(self, config, model_fn, model_train_batch_size=256, 
                model_train_epoch=75, exploration=0.9, 
                exploration_decay_rate=0.8, opt_metric='val_loss', 
                opt='max', starting_tol=0.01, tol_decay=0.5):

        self._env = CustomEnvironment(config, model_train_epoch=model_train_epoch,
                                    model_train_batch_size=model_train_batch_size, 
                                    input_model=model_fn, opt_metric=opt_metric, opt=opt, 
                                    starting_tol=starting_tol, tol_decay=tol_decay)
        self._agent = DeepQNetwork(states=self._env.states(), actions=self._env.actions(),
                     max_episode_timesteps=self._env.max_episode_timesteps(),
                     memory=60, batch_size=1, exploration=dict(type='decaying', unit='timesteps', decay='exponential',
                                                            initial_value=exploration, decay_steps=100, decay_rate=exploration_decay_rate)
                     )

        self.runner = Runner(agent=self._agent, environment=self._env)

    def set_k_folds(self, n_folds, pick_random=None):
        self._env.set_k_folds(n_folds, pick_random)

    def train_on_data(self, X_train, y_train, X_valid=None, y_valid=None):
        self._env.train_on_data(X_train, y_train, X_valid, y_valid)

    def run(self, num_episodes):
        self._env.reset_history()
        self._env.set_num_episodes(num_episodes)
        self.runner.run(num_episodes=num_episodes)
        print(self._env.get_history())
        self.runner.close()