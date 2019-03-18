import gym
from wrap_env import wrap_env
import numpy as np
import gym_boxworld
import matplotlib.pyplot as plt
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, wrap_boxworld


env_name = 'BoxWorld'
env_id = env_name + 'NoFrameskip-v4'
print(env_id)
env = make_atari(env_id)
# if 'BoxWorld' in env_id:
#     print('using wrap_boxworld!')
#     env = wrap_boxworld(env, episode_life=False, clip_rewards=True, frame_stack=True, scale=True)
# else:
#     env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=True, scale=False)


observation = env.reset()
print(env.action_space)
i = 0
for i in range(10000):
    observation, reward, done, info = env.step(int(input("input")))
    if done:
        observation = env.reset()
    print(reward, done, info)
    # observation = np.array(observation)
    # print(observation.shape)
    # img = observation[:, :, 0]
    # fig = plt.figure(2)
    # plt.clf()
    # plt.imshow(img)
    # # fig.canvas.draw()
    # plt.pause(0.0001)
    env.render()


# env = gym.make('BreakoutNoFrameskip-v4')
# # env = gym.make("{}NoFrameskip-v4".format('Breakout'))   # 定义使用 gym 库中的那一个环境
# env = wrap_env(env)
# obervation = env.reset()
#
# for i in range(10000):
#     obervation = env.step(env.action_space.sample())[0][:, :, 0]
#     # obervation = np.array(obervation._frames)
#     # obervation = np.reshape(obervation, [84, 84, -1])[:, :, 0]
#     print(obervation.shape)
#     img = obervation
#     fig = plt.figure(2)
#     plt.clf()
#     plt.imshow(img / 255)
#     # fig.canvas.draw()
#     plt.pause(0.0001)
#     # env.render()
