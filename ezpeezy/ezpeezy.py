from tensorforce.execution import Runner
from tensorforce.agents import DeepQNetwork
from .environment import CustomEnvironment

class Ezpeezy():
    def __init__(self, config, model_fn, opt='max', starting_tol=0.01, tol_decay=0.8):
        env = CustomEnvironment(config, input_model=model_fn, opt=opt, 
                                starting_tol=starting_tol, tol_decay=tol_decay)
        agent = DeepQNetwork(states=env.states(), actions=env.actions(),
                     max_episode_timesteps=env.max_episode_timesteps(),
                     memory=60, batch_size=1, exploration=dict(type='decaying', unit='timesteps', decay='exponential',
                                                            initial_value=0.9, decay_steps=1000, decay_rate=0.8)
                     )

        self.runner = Runner(agent=agent, environment=env)

    def run(self, num_episodes):
        self.runner.run(num_episodes=num_episodes)
        self.runner.close()