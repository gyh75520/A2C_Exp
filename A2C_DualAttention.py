import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.a2c.utils import conv, linear, batch_to_seq, seq_to_batch, lstm, ortho_init
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C
from stable_baselines.common.atari_wrappers import make_atari
from utils import custom_cnn, MHDPA, linear_without_bias
'''
self attention + attention
'''


class DualAttentionLstmPolicy(LstmPolicy):
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
            last_num_height = extracted_features.shape[1]
            last_num_width = extracted_features.shape[2]
            # print(last_width)
            last_num_features = extracted_features.shape[3]
            n_hiddens = 42

            # [B,H*W,num_heads,Deepth]
            num_heads = 2
            MHDPA_output = MHDPA(extracted_features, "extracted_features", num_heads=num_heads)
            MHDPA_output = tf.reshape(MHDPA_output, [-1, last_num_height, last_num_width, num_heads * last_num_features])
            print('MHDPA_output', MHDPA_output)

            x2 = tf.reshape(extracted_features, [-1, last_num_height * last_num_width, last_num_features])
            print('x2', x2)
            x3 = tf.nn.relu(conv(extracted_features, 'x3', n_filters=n_hiddens, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
            print('x3', x3)

            print('states', self.states_ph)
            # ob = [envs,steps] -- rnn_state = [envs]*steps
            h0 = tf.expand_dims(self.states_ph, 1)
            h0 = tf.tile(h0, [1, self.n_steps, 1])
            print('h0', h0)
            h0 = tf.reshape(h0, [-1, h0.shape[2]])
            print('h0', h0)
            h1 = linear_without_bias(h0, 'fc_h1', n_hidden=n_hiddens, init_scale=np.sqrt(2))
            print('h1', h1)
            # replicate [1,n_hiddens] to [1,22*16,n_hiddens]
            h2 = tf.expand_dims(h1, 1)
            h2 = tf.tile(h2, [1, last_num_height * last_num_width, 1])
            print('h2', h2)

            h3 = tf.reshape(h2, [-1, last_num_height, last_num_width, n_hiddens])
            print('h3', h3)

            a1 = tf.nn.tanh(tf.add(h3, x3))
            a2 = tf.nn.relu(conv(a1, 'a2', n_filters=1, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
            print('a2', a2)

            a3 = tf.nn.softmax(tf.reshape(a2, [-1, last_num_height * last_num_width]))  # attetion
            print('a3', a3)
            self.attention = a3

            a4 = tf.expand_dims(a3, 2)
            a4 = tf.tile(a4, [1, 1, last_num_features])
            print('a4', a4)

            context = tf.reduce_sum(tf.multiply(a4, x2), 2)
            print('context', context)
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
    model = A2C(DualAttentionLstmPolicy, env, verbose=1)
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
