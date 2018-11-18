import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt


def subsample(t, vt, bins):
    """Given a data such that value vt[i] was observed at time t[i],
    group it into bins: (bins[j], bins[j+1]) such that values
    for bin j is equal to average of all vt[k], such that
    bin[j] <= t[k] < bin[j+1].

    Parameters
    ----------
    t: np.array
        times at which the values are observed
    vt: np.array
        values for those times
    bins: np.array
        endpoints of the bins.
        for n bins it shall be of length n + 1.

    Returns
    -------
    x: np.array
        endspoints of all the bins
    y: np.array
        average values in all bins"""
    bin_idx = np.digitize(t, bins) - 1
    print(bin_idx)
    v_sums = np.zeros(len(bins), dtype=np.float32)
    v_cnts = np.zeros(len(bins), dtype=np.float32)
    print(len(v_sums), len(bin_idx), len(vt))
    np.add.at(v_sums, bin_idx, vt)
    np.add.at(v_cnts, bin_idx, 1)

    # ensure graph has no holes
    zs = np.where(v_cnts == 0)
    # assert v_cnts[0] > 0
    for zero_idx in zs:
        v_sums[zero_idx] = v_sums[zero_idx - 1]
        v_cnts[zero_idx] = v_cnts[zero_idx - 1]

    return bins, v_sums / v_cnts


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


def tsplot_result(log_dir, title='Learning Curve'):
    import seaborn as sns
    import pandas as pd
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y = movingAverage(y, window=100)
    print(len(y))
    x = x[len(x) - len(y):]
    print(len(x))
    x, y = subsample(t=x, vt=y, bins=np.linspace(0, 20000, 100 + 1))
    data = pd.DataFrame({'Frame': x,  'Average Episode Reward': y})
    sns.tsplot(data=[y])


if __name__ == '__main__':
    # plot_result(log_dir='test/')
    tsplot_result(log_dir='test/')
    # print(np.cumsum(load_results('test/').l))
    # from stable_baselines.results_plotter import plot_results
    # plot_results(['test/', 'test/CartPole/'], int(10e6), 'timesteps', 'Learning Curve')
    plt.show()
    print('end')
