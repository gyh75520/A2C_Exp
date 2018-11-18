from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os
import numpy as np
from stable_baselines import A2C
from A2C_attention import AttentionPolicy, LstmPolicy

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


def make_env(env_id, rank, log_dir, seed=0):
    def _init():
        env = make_atari(env_id)
        env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def run(model_name, env_name, num_cpu, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env_id = env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)

    if model_name == 'A2C_Attention':
        model = A2C(AttentionPolicy, env, verbose=1)
    elif model_name == 'A2C':
        model = A2C(LstmPolicy, env, verbose=1)
    else:
        model = None
    # model.learn(total_timesteps=int(1e7), callback=callback)
    model.learn(total_timesteps=int(1e7))
    model.save(log_dir + model_name + '_' + env_name)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
env_name = 'Seaquest'
num_cpu = 4
A2C_Attention_log_dir = 'attention_exp/A2C_Attention_{}/'.format(env_name)
A2C_log_dir = 'attention_exp/A2C_{}/'.format(env_name)
# run('A2C_Attention', env_name, num_cpu, A2C_Attention_log_dir)
run('A2C', env_name, num_cpu, A2C_log_dir)
print('finish')
