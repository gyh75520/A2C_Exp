import gym
from wrap_env import wrap_env
import numpy as np
import gym_boxworld
import matplotlib.pyplot as plt

env = gym.make('BoxWorldNoFrameskip-v4')
# env = gym.make("{}NoFrameskip-v4".format('Breakout'))   # 定义使用 gym 库中的那一个环境
env = wrap_env(env)
env.verbose = True
obervation = env.reset()
# print(np.where(obervation[0] > 0)[0])
print(obervation.shape)
i = 0
for i in range(10000):
    obervation = env.step(env.action_space.sample())[0]
    # print(obervation.shape)
    # index = np.array(np.where(obervation[0] > 0))
    # print(index[0][0], index[1][0], index[0][0])
    # print(obervation[5][36][2])
    img = obervation
    fig = plt.figure(1)
    plt.clf()
    plt.imshow(img / 255)
    fig.canvas.draw()
    plt.pause(0.0001)
    env.render()
input('input')
