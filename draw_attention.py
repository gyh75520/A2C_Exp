from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.tile_images import tile_images
import os
import numpy as np
from stable_baselines import A2C
from A2C_attention import AttentionPolicy, LstmPolicy
from A2C_attention2 import Attention2Policy
import cv2


def make_env(env_id, rank, log_dir, seed=0):
    def _init():
        env = make_atari(env_id)
        env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def attention_render(model_name, env_name, num_cpu, log_dir):
    if not os.path.exists(log_dir):
        raise ('log_dir not Exists')

    env_id = env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)

    if model_name == 'A2C_Attention':
        model = A2C(AttentionPolicy, env, verbose=1, tensorboard_log=log_dir + 'tensorboard/')
    elif model_name == 'A2C_Attention2':
        model = A2C(Attention2Policy, env, verbose=1, tensorboard_log=log_dir + 'tensorboard/')
    elif model_name == 'A2C':
        model = A2C(LstmPolicy, env, verbose=1, tensorboard_log=log_dir + 'tensorboard/')
    else:
        model = None
    model = model.load(log_dir + model_name + '_' + env_name, env=env)

    obs = env.reset()
    # print(env.observation_space)
    # cv2.imshow('test', RGB2BGR(obs[0]))
    # cv2.waitKey(0)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        attentions = model.get_attention(obs, _states, done)[0]
        attentions_img = []
        # print('attention', np.array(attention).shape)
        for i, attention in enumerate(attentions):
            attention = np.array(attention)
            attention = np.reshape(attention, [env.observation_space.shape[0] // 10, env.observation_space.shape[1] // 10, 1])
            attention = np.repeat(attention, [10] * attention.shape[0], axis=0)
            attention = np.repeat(attention, [10] * attention.shape[1], axis=1)
            attention = attention * 255
            attentions_img.append(attention)
            # print(np.sum(attention))
        attentions = tile_images(attentions_img)
        cv2.imshow('attention', attentions)
        cv2.waitKey(1)
        # break
        env.render()
    return model


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
env_name = 'Breakout'
num_cpu = 4
A2C_Attention2_log_dir = 'attention_exp/A2C_Attention2_{}/'.format(env_name)

attention_render('A2C_Attention2', env_name, num_cpu, A2C_Attention2_log_dir)
