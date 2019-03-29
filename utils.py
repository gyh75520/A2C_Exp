import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, ortho_init


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_3', layer_3)
    # layer_3 = conv_to_fc(layer_3)

    # return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return layer_3


def custom_cnn(scaled_images, **kwargs):
    """
    output h/10 w/10.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=5, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_3', layer_3)
    # layer_3 = conv_to_fc(layer_3)

    # return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return layer_3


def linear_without_bias(input_tensor, scope, n_hidden, init_scale=1.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        # bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        # return tf.matmul(input_tensor, weight) + bias
        return tf.matmul(input_tensor, weight)


def layerNorm(input_tensor, scope, eps=1e-5):
    """
    Creates a layerNormalization module for TensorFlow
    ref:https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/layer_norm.py

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [batch_size,layer_dim]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.get_shape()[1].value
        gamma = tf.get_variable("gamma", [hidden_size], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [hidden_size], initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [1], keep_dims=True)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def get_coor(input_tensor):
    """
    The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :return: (TensorFlow Tensor) [B,Height,W,2]
    """
    batch_size = input_tensor.get_shape()[0].value
    height = input_tensor.get_shape()[1].value
    width = input_tensor.get_shape()[2].value
    coor = []
    for h in range(height):
        w_channel = []
        for w in range(width):
            w_channel.append([float(h / height), float(w / width)])
        coor.append(w_channel)

    coor = tf.expand_dims(tf.constant(coor, dtype=input_tensor.dtype), axis=0)
    # [1,Height,W,2] --> [B,Height,W,2]
    coor = tf.tile(coor, [batch_size, 1, 1, 1])
    return coor


def MHDPA(input_tensor, scope, num_heads):
    """
    An implementation of the Multi-Head Dot-Product Attention architecture in "Relational Deep Reinforcement Learning"
    https://arxiv.org/abs/1806.01830
    ref to the RMC architecture on https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :param scope: (str) The TensorFlow variable scope
    :param num_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,Height*W,num_heads,D]
    """
    with tf.variable_scope(scope):
        last_num_height = input_tensor.get_shape()[1].value
        last_num_width = input_tensor.get_shape()[2].value
        last_num_features = input_tensor.get_shape()[3].value

        key_size = value_size = last_num_features
        qkv_size = 2 * key_size + value_size
        # total_size Denoted as F, num_heads Denoted as H
        total_size = qkv_size * num_heads

        # Denote N = last_num_height * last_num_width
        # [B*N,Deepth]
        extracted_features_reshape = tf.reshape(input_tensor, [-1, last_num_features])
        # [B*N,F]
        qkv = linear(extracted_features_reshape, "QKV", total_size)
        # [B*N,F]
        qkv = layerNorm(qkv, "qkv_layerNorm")
        # [B,N,F]
        qkv = tf.reshape(qkv, [-1, last_num_height * last_num_width, total_size])
        # [B,N,H,F/H]
        qkv_reshape = tf.reshape(qkv, [-1, last_num_height * last_num_width, num_heads, qkv_size])
        print("qkv_reshape", qkv_reshape)
        # [B,N,H,F/H] -> [B,H,N,F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        print("qkv_transpose", qkv_transpose)
        q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

        q *= qkv_size ** -0.5
        # [B,H,N,N]
        dot_product = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(dot_product)
        # [B,H,N,V]
        output = tf.matmul(weights, v)
        # [B,H,N,V] -> [B,N,H,V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        return output_transpose, weights


def residual_block(x, y):
    """
    Z = W*y + x
    :param x: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :param y: (TensorFlow Tensor) The input tensor from MHDPA [B,Height*W,num_heads,D]
    :return: (TensorFlow Tensor) [B,Height*W,num_heads,D]
    """
    last_num_height = x.get_shape()[1].value
    last_num_width = x.get_shape()[2].value
    last_num_features = x.get_shape()[3].value
    # W*y
    y_Matmul_W = conv(y, 'y_Matmul_W', n_filters=last_num_features, filter_size=1, stride=1, init_scale=np.sqrt(2))
    print('y_Matmul_W', y_Matmul_W)
    # [B,Height,W,D] --> [B,Height*W,D]
    x_reshape = tf.reshape(x, [-1, last_num_width * last_num_height, last_num_features])
    x_edims = tf.expand_dims(x_reshape, axis=2)
    num_heads = y.get_shape()[2]
    # [B,Height,W,D] --> [B,Height*W,H,D]
    x_edims = tf.tile(x_edims, [1,  1, num_heads, 1])
    # W*y + x
    residual_output = tf.add(y_Matmul_W, x_edims)
    return residual_output

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

    v_sums = np.zeros(len(bins), dtype=np.float32)
    v_cnts = np.zeros(len(bins), dtype=np.float32)

    # np.add.at(v_sums, bin_idx, vt)
    # fix np.add.at(v_sums, bin_idx, vt)
    for index, t_iter in enumerate(t):
        binIndex = np.digitize(t_iter, bins) - 1
        np.add.at(v_sums, [binIndex], vt[index])
    np.add.at(v_cnts, bin_idx, 1)

    # ensure graph has no holes
    zs = np.where(v_cnts == 0)[0]

    # v_sums = np.delete(v_sums, zs)
    # v_cnts = np.delete(v_cnts, zs)
    # bins = np.delete(bins, zs)

    # assert v_cnts[0] > 0
    for zero_idx in zs:
        v_sums[zero_idx] = v_sums[zero_idx - 1]
        v_cnts[zero_idx] = v_cnts[zero_idx - 1]
    zs = np.where(v_cnts == 0)[0]
    print('If zs is not Null,v_cnts have zeros', zs)
    return bins[1:], (v_sums / (v_cnts + int(1e-7)))[1:]


def movingAverage(values, window):
    # smooth valus by doing a moving Average
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_result(log_dir, title='Learning Curve'):
    # print(np.cumsum(load_results(log_dir).1.values))
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    # print(x, y)
    # y = movingAverage(y, window=100)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + ' Smoothed')
    plt.show()


def tsplot_result(log_dirs_dict, num_timesteps, title='Learning Curve'):
    # print('load_results', load_results(log_dir))
    import seaborn as sns
    import pandas as pd
    datas = []
    for key in log_dirs_dict:
        log_dirs = log_dirs_dict[key]
        for index, dir in enumerate(log_dirs):
            init_data = load_results(dir)
            init_data = init_data[init_data.l.cumsum() <= num_timesteps]
            x, y = ts2xy(init_data, 'timesteps')
            y = movingAverage(y, window=100)
            x = x[len(x) - len(y):]
            # x = x[:len(y)]
            print('y', y)
            x, y = subsample(t=x, vt=y, bins=np.linspace(0, num_timesteps, int(1000) + 1))
            x = np.append(x, np.array([0]))
            y = np.append(y, np.array([0]))
            print('y after subsample', y)

            # y = movingAverage(y, window=10)
            # # x = x[len(x) - len(y):]
            # x = x[:len(y)]
            data = pd.DataFrame({'Timesteps': x,  'Reward': y, 'subject': np.repeat(index, len(x)), 'Algorithm': np.repeat(key, len(x))})

            datas.append(data)

    data_df = pd.concat(datas, ignore_index=True)

    print('data', data_df)
    sns.tsplot(data=data_df, time='Timesteps', value='Reward', unit='subject', condition='Algorithm')


if __name__ == '__main__':
    # plot_result(log_dir='test/')
    # import seaborn as sns
    # gammas = sns.load_dataset("gammas")
    # #
    # print('gammas', gammas)
    # ax = sns.tsplot(time="timepoint", value="BOLD signal", unit="subject", condition="ROI", data=gammas)

    # tsplot_result(log_dirs=['test/CartPole/', 'test/dqn_breakout/'])

    # print(np.cumsum(load_results('test/').l))P
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
    algs = ['A2C', 'A2C_SelfAttention']
    env_name = 'BoxWorld'
    log_dirs = {}
    for alg in algs:
        log_dirs[alg] = ['attention_exp/{}/{}_0'.format(alg, env_name)]
    # log_dirs = {'A2C_Attention': ['attention_exp1/A2C_Attention_Qbert1'], 'A2C': ['attention_exp1/A2C_Qbert', 'attention_exp1/A2C_Qbert1']}
    tsplot_result(log_dirs_dict=log_dirs, num_timesteps=int(1e7))
    plt.show()
    print('end')
