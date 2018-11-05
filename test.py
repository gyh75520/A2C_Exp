'''
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda:env])


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines import A2C

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if(n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100])

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print('Saving new best model')
                _locals['self'].save(log_dir + 'best_model.pkl')

            print(x[-1], 'timesteps')
            print('best_mean_reward:{:.2f} - last mean reward per episode:{:2f}'.format(best_mean_reward, mean_reward))

    n_steps += 1
    return False


log_dir = 'test/CartPole/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# def make_env(env_id, rank, seed=0):
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         return env
#
#     # set_global_seeds(seed)
#     return _init
#
#
# env_id = 'CartPole-v1'
# num_cpu = 4
# env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

env = gym.make("CartPole-v1")
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])


model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1)
model.learn(total_timesteps=30000, callback=callback)

# model.save('a2c_lunar')
#
# del model
#
# model = A2C.load('a2c_lunar')

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print('info:', info)
    env.render()
