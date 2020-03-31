import matplotlib.pyplot as plt

class Visualizer():

    @staticmethod
    def plot_history(history, max_num_episodes, y_label):
        color_variations = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for episode in history['episode'].unique():
            episode_subset = history.loc[history['episode'] == episode]
            plt.plot(episode_subset.index, episode_subset[y_label], 
                    color=color_variations[int(episode) % len(color_variations)])
        plt.ylabel(y_label)
        plt.xlabel('absolute_time_steps')
        plt.xticks(range(len(history)))
        plt.show()