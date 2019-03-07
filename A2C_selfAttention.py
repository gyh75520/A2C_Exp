import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.a2c.utils import conv, linear, batch_to_seq, seq_to_batch, lstm, ortho_init
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C
from stable_baselines.common.atari_wrappers import make_atari
'''
using attention as intrinsic rewards
'''


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

        return output_transpose


class SelfAttentionLstmPolicy(LstmPolicy):
    __module__ = None

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 cnn_extractor=custom_cnn, layer_norm=False, feature_extraction="cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            extracted_features = cnn_extractor(self.processed_x, **kwargs)  # # [B,H,W,Deepth]
            print('extracted_features', extracted_features)

            # [B,H*W,num_heads,Deepth]
            MHDPA_output = MHDPA(extracted_features, "extracted_features", num_heads=2)
            print(MHDPA_output)
            input_sequence = batch_to_seq(MHDPA_output, self.n_env, n_steps)
            # input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)

            rnn_output = seq_to_batch(rnn_output)
            # print('rnn_output', rnn_output, '      snew', self.snew)

            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

    def get_attention(self, obs, state=None, mask=None):
        return self.sess.run([self.attention],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def step_with_attention(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp, self.attention],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        else:
            return self.sess.run([self.action, self._value, self.snew, self.neglogp, self.attention],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


def make_env(env_id, rank, seed=0):
    def _init():
        env = make_atari(env_id)
        # env = VecFrameStack(env, n_stack=4)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def RGB2BGR(img):
    # opencv读取图片的默认像素排列是BGR
    return img[..., ::-1]


if __name__ == '__main__':
    env_id = 'BreakoutNoFrameskip-v4'
    num_cpu = 1

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = VecFrameStack(env, n_stack=4)
    # print(env.observation_space.shape)
    model = A2C(SelfAttentionLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=1000)
    # model.save("A2C_Attention_breakout")
    # del model
    # model = A2C.load("A2C_Attention_breakout", env=env, policy=Attention2Policy)

    import cv2
    obs = env.reset()
    # print(env.observation_space)
    # cv2.imshow('test', RGB2BGR(obs[0]))
    # cv2.waitKey(0)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        attention = model.get_attention(obs, _states, done)[0]
        attention = np.array(attention)
        attention = np.reshape(attention, [env.observation_space.shape[0] // 10, env.observation_space.shape[1] // 10])
        attention = np.repeat(attention, [10] * attention.shape[0], axis=0)
        attention = np.repeat(attention, [10] * attention.shape[1], axis=1)
        attention = attention * 255
        print(np.sum(attention))
        cv2.imshow('attention', attention)
        cv2.waitKey(1)
        # break
        env.render()
