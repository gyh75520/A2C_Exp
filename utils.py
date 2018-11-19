import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt


# def subsample(t, vt, bins):
#     """Given a data such that value vt[i] was observed at time t[i],
#     group it into bins: (bins[j], bins[j+1]) such that values
#     for bin j is equal to average of all vt[k], such that
#     bin[j] <= t[k] < bin[j+1].
#
#     Parameters
#     ----------
#     t: np.array
#         times at which the values are observed
#     vt: np.array
#         values for those times
#     bins: np.array
#         endpoints of the bins.
#         for n bins it shall be of length n + 1.
#
#     Returns
#     -------
#     x: np.array
#         endspoints of all the bins
#     y: np.array
#         average values in all bins"""
#     bin_idx = np.digitize(t, bins) - 1
#     print(bin_idx)
#     v_sums = np.zeros(len(bins), dtype=np.float32)
#     v_cnts = np.zeros(len(bins), dtype=np.float32)
#     print(len(v_sums), len(bin_idx), len(vt))
#     np.add.at(v_sums, bin_idx, vt)
#     np.add.at(v_cnts, bin_idx, 1)
#
#     # ensure graph has no holes
#     zs = np.where(v_cnts == 0)
#     # assert v_cnts[0] > 0
#     for zero_idx in zs:
#         v_sums[zero_idx] = v_sums[zero_idx - 1]
#         v_cnts[zero_idx] = v_cnts[zero_idx - 1]
#
#     return bins, v_sums / v_cnts

def subsample(t, vt, bins):
    """
    Given a data such that value vt[i] was observed at time t[i],
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
        average values in all bins
    """
    bin_idx = np.digitize(t, bins) - 1
    print('t', t)
    print('bin_idx', bin_idx)
    v_sums = np.zeros(len(bins), dtype=np.float32)
    v_cnts = np.zeros(len(bins), dtype=np.float32)
    np.add.at(v_sums, bin_idx, vt)
    np.add.at(v_cnts, bin_idx, 1)
    # ensure graph has no holes
    zs = np.where(v_cnts == 0)
    # assert v_cnts[0] > 0
    for zero_idx in zs:
        v_sums[zero_idx] = v_sums[zero_idx - 1]
        v_cnts[zero_idx] = v_cnts[zero_idx - 1]

    print('v_sums', v_sums)
    return bins[1:], (v_sums / (v_cnts + 1e-7))[1:]


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


def tsplot_result(log_dirs, title='Learning Curve'):
    # print('load_results', load_results(log_dir))
    import seaborn as sns
    import pandas as pd
    xlist = []
    ylist = []
    datas = []
    for index, dir in enumerate(log_dirs):

        x, y = ts2xy(load_results(dir), 'timesteps')

        if dir == 'test/dqn_breakout/':
            y += 10
        print(y.shape)
        y = movingAverage(y, window=100)
        x = x[len(x) - len(y):]
        x, y = subsample(t=x, vt=y, bins=np.linspace(1000, int(6000), int(50) + 1))

        data = pd.DataFrame({'Frame': x,  'Reward': y, 'subject': np.repeat(index, len(x))})
        xlist.append(x)
        ylist.append(y)
        datas.append(data)

    # data = pd.DataFrame({'Frame': x,  'Average Episode Reward': ylist})
    data_df = pd.concat(datas, ignore_index=True)

    # data = pd.DataFrame(ylist)
    print('data', data_df)
    sns.tsplot(data=data_df, time='Frame', value='Reward', unit='subject')


if __name__ == '__main__':
    # plot_result(log_dir='test/')
    import seaborn as sns
    gammas = sns.load_dataset("gammas")

    print('gammas', gammas)
    # ax = sns.tsplot(time="timepoint", value="BOLD signal", unit="subject", condition="ROI", data=gammas)

    tsplot_result(log_dirs=['test/CartPole/', 'test/dqn_breakout/'])

    # print(np.cumsum(load_results('test/').l))
    # from stable_baselines.results_plotter import plot_results
    # plot_results(['test/', 'test/CartPole/'], int(10e6), 'timesteps', 'Learning Curve')

    # import seaborn as sns
    # import pandas as pd
    # x = np.linspace(0, 15, 31)
    # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    # print(data.shape)
    # print(pd.DataFrame(data).head())  # 每一行数据是一个变量，31列是代表有31天或31种情况下的观测值。
    # # 创建数
    #
    # sns.tsplot(data=data,
    #            err_style="ci_band",   # 误差数据风格，可选：ci_band, ci_bars, boot_traces, boot_kde, unit_traces, unit_points
    #            interpolate=True,      # 是否连线
    #            # ci=[40, 70, 90],       # 设置误差 置信区间
    #            color='g'            # 设置颜色
    #            )
    plt.show()
    print('end')
