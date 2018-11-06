from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import CnnPolicy
from DQN_PiecewiseSchedule import DQN
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


log_dir = 'test_schedules/dqn_breakout/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

env = make_atari('BreakoutNoFrameskip-v4')
# env = VecFrameStack(env, n_stack=4)
env = Monitor(env, log_dir, allow_early_resets=True)

model = DQN(env=env,
        policy=CnnPolicy,
        learning_rate=1e-4,
        buffer_size=1000000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=50000,
        target_network_update_freq=10000,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir+'tensorboard/',
        )
model.learn(total_timesteps=int(1e8), callback=callback)
model.save("dqn_breakout")

# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
