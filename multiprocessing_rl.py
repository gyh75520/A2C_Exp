import time
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def evaluate(model, env, num_steps=2000):
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
        actions, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(actions)
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

    mean_rewards = [[0.0] for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards = np.mean(episode_rewards[i])
        n_episodes += len(episode_rewards[i])

    print('Mean Rewards:', mean_rewards, ' Num episode:', n_episodes)

    return mean_rewards


env_id = 'CartPole-v1'
num_cpu = 4

env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# print(env.num_envs)
model = A2C(MlpPolicy, env, verbose=0)

mean_reward_before_train = evaluate(model, env)

n_timesteps = 300000
start_time = time.time()
model.learn(n_timesteps)
total_time = time.time() - start_time

print('Took {:.2f}s for multiprocessed version - {:.2f} FPS'.format(total_time, n_timesteps / total_time))
# mean_reward_after_train = evaluate(model, env)


env = DummyVecEnv([lambda:gym.make(env_id)])
single_process_model = A2C(MlpPolicy, env, verbose=0)

start_time = time.time()
single_process_model.learn(n_timesteps)
total_time = time.time() - start_time

mean_reward_after_train = evaluate(model, env)

print('Took {:.2f}s for singleprocessed version - {:.2f} FPS'.format(total_time, n_timesteps / total_time))
mean_reward_after_train = evaluate(single_process_model, env)
