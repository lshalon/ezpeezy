import matplotlib.pyplot as plt

class Visualizer():
    """
    This class is used to represent all of the visualizations neccessary
    to show the user.

    Methods
    -------
    plot_history(max_num_episodes, y_label)
        Plots the history with absolute time steps on the x-axis and the monitored
        metric on the y-axis. Different episodes have different coloring.
    """

    @staticmethod
    def plot_history(history, max_num_episodes, y_label):
        """
        Plots the history with absolute time steps on the x-axis and the monitored
        metric on the y-axis. Different episodes have different coloring.

        Parameters
        ----------
        max_num_episodes : int
            the maximum number of episodes in the current run
        y_label : string
            the monitored metric's name

        Outputs
        -------
        graph of the absolute timesteps and the monitored metric on the y-axis.
        Different episodes will have different color configurations.
        """
        color_variations = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for episode in history['episode'].unique():
            episode_subset = history.loc[history['episode'] == episode]
            plt.plot(episode_subset.index, episode_subset[y_label], 
                    color=color_variations[int(episode) % len(color_variations)])
        plt.ylabel(y_label)
        plt.xlabel('absolute_time_steps')
        plt.xticks(range(len(history)))
        plt.show()