import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt


def movingAverage(values, window):
    # smooth valus by doing a moving Average
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_result(log_dir, title='Learning Curve'):
    # print(np.cumsum(load_results(log_dir).1.values))
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    # print(x, y)
    y = movingAverage(y, window=100)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + ' Smoothed')
    plt.show()


if __name__ == '__main__':
    # plot_result(log_dir='test/')
    # print(np.cumsum(load_results('test/').l))
    from stable_baselines.results_plotter import plot_results
    plot_results(['test/', 'test/CartPole/'], int(10e6), 'timesteps', 'Learning Curve')
    plt.show()
    print('end')
