from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os
import numpy as np

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if(n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 10:
            mean_reward = np.mean(y[-10])

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print('Saving new best model')
                _locals['self'].save(log_dir + 'best_model.pkl')

            print(x[-1], 'timesteps')
            print('best_mean_reward:{:.2f} - last mean reward per episode:{:2f}'.format(best_mean_reward, mean_reward))

    n_steps += 1
    return False


log_dir = 'test/dqn_breakout/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = make_atari('BreakoutNoFrameskip-v4')
# env = VecFrameStack(env, n_stack=4)
env = Monitor(env, log_dir, allow_early_resets=True)

from stable_baselines.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: env])

model = DQN(CnnPolicy, env, verbose=0)

print('init')
model.learn(total_timesteps=25000, callback=callback)
model.save("dqn_breakout")

# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
