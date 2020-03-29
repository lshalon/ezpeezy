from tensorforce.execution import Runner
from tensorforce.agents import DeepQNetwork
from .environment import CustomEnvironment

class Ezpeezy():
    def __init__(self, model_fn):
        env = CustomEnvironment({'dropout_rate': '0:0.9', 'max_pool_size': 'i2:5'}, input_model=model_fn, opt='min')
        agent = DeepQNetwork(states=env.states(), actions=env.actions(),
                     max_episode_timesteps=env.max_episode_timesteps(),
                     memory=60, batch_size=1, exploration=dict(type='decaying', unit='timesteps', decay='exponential',
                                                            initial_value=0.9, decay_steps=1000, decay_rate=0.8)
                     )

        runner = Runner(agent=agent, environment=env)
        runner.run(num_episodes=100)
        runner.close()