from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, wrap_boxworld
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os
import numpy as np
from stable_baselines import A2C
from stable_baselines.common.policies import CnnLstmPolicy
from A2C_attention import AttentionPolicy
from A2C_attention2 import Attention2Policy
from A2C_attention3 import Attention3Policy
from A2C_attention4 import Attention4Policy
from A2C_selfAttention import SelfAttentionLstmPolicy
from A2C_DualAttention import DualAttentionLstmPolicy
from A2C_selfAttention_cin import SelfAttentionCinLstmPolicy
import gym_boxworld


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


def make_env(env_id, rank, log_dir, useMonitor=True, seed=0):
    def _init():
        env = make_atari(env_id)
        # if 'BoxWorld' in env_id:
        #     print('using wrap_boxworld!')
        #     env = wrap_boxworld(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False)
        # else:
        #     env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False)
        if useMonitor:
            env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def get_model(model_name, env, log_dir):
    if model_name == "A2C_DualAttention":
        model = A2C(DualAttentionLstmPolicy, env, verbose=1)
    elif model_name == "A2C_SelfAttention_Cin":
        model = A2C(SelfAttentionCinLstmPolicy, env, verbose=1)
    elif model_name == "A2C_SelfAttention":
        model = A2C(SelfAttentionLstmPolicy, env, verbose=1)
    elif model_name == 'A2C_Attention':
        model = A2C(AttentionPolicy, env, verbose=1, tensorboard_log=log_dir + 'tensorboard/')
    elif model_name == 'A2C_Attention2':
        model = A2C(Attention2Policy, env, verbose=1, tensorboard_log=log_dir + 'tensorboard/')
    elif model_name == 'A2C_Attention3':
        model = A2C(Attention3Policy, env, verbose=1)
    elif model_name == 'A2C_Attention4':
        model = A2C(Attention4Policy, env, verbose=1)
    elif model_name == 'A2C':
        model = A2C(CnnLstmPolicy, env, verbose=1)
    else:
        raise('{} Not Exist'.format(model_name))
    return model


def run(model_name, env_name, num_cpu, log_dir):
    # if log_dir exists,auto add new dir by order
    while os.path.exists(log_dir):
        lastdir_name = log_dir.split('/')[-2]
        times = int(lastdir_name.split('_')[-1])
        old = '_{}'.format(times)
        new = '_{}'.format(times + 1)
        log_dir = log_dir.replace(old, new)
    os.makedirs(log_dir)

    print(("---------------Create dir:{} Successful!-------------\n").format(log_dir))

    env_id = env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)
    model = get_model(model_name, env, log_dir)

    total_timesteps = int(1e7)
    print(("---------------Modle:{} num_cpu:{} total_timesteps:{} Start to train!-------------").format(model_name, num_cpu, total_timesteps))

    # model.learn(total_timesteps=int(1e7), callback=callback)
    model.learn(total_timesteps=total_timesteps)
    model.save(log_dir + model_name + '_' + env_name)


def test(model_name, env_name, num_cpu, log_dir):
    env_id = env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, i, log_dir, useMonitor=False) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)
    model = get_model(model_name, env, log_dir)

    model = model.load(log_dir + model_name + '_' + env_name, env=env)

    obs = env.reset()
    from matplotlib import pyplot as plt
    show_num = 1
    while True:
        action, _states = model.predict(obs)
        # obs, rewards, done, info = env.step([int(input('action:'))]*num_cpu)
        obs, rewards, done, info = env.step(action)
        img = obs[show_num, :, :, :]
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img / 255)
        fig.canvas.draw()

        if 'SelfAttention' in model_name and 'Box' in env_name and 'World' in env_name:
            agent_position = env.env_method('get_current_agent_position')[show_num]
            print('agent_position', agent_position)
            attention = model.get_attention(obs, _states, done)[0]
            # head_0
            attention0 = attention[show_num][0][agent_position]

            left = [attention0[agent_position - 14] if agent_position - 14 >= 0 else 0]
            right = [attention0[agent_position + 14] if agent_position + 14 > 155 else 0]
            attention0_udlr = [left, right, attention0[agent_position - 1], attention0[agent_position + 1]]
            # print('top :{} left:{}'.format(attention0[agent_position-14],attention0[agent_position-1]))
            attention0 = np.reshape(attention0, [14, 14])
            fig = plt.figure(2)
            plt.clf()
            plt.imshow(attention0, cmap='gray')
            fig.canvas.draw()

            # head_1
            attention1 = attention[show_num][1][agent_position]

            left = [attention1[agent_position - 14] if agent_position - 14 >= 0 else 0]
            right = [attention1[agent_position + 14] if agent_position + 14 > 155 else 0]
            attention1_udlr = [left, right, attention1[agent_position - 1], attention1[agent_position + 1]]

            attention1 = np.reshape(attention1, [14, 14])
            fig = plt.figure(3)
            plt.clf()
            plt.imshow(attention1, cmap='gray')
            fig.canvas.draw()

            print(action[show_num], np.argmax(attention0_udlr), np.argmax(attention1_udlr), "max attention:", np.max(attention0_udlr), np.max(attention1_udlr))
        # env.render()
        plt.pause(0.000001)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
env_name = 'BoxRandWorld'
num_cpu = 4
A2C_SelfAttention_Cin_log_dir = 'attention_exp/A2C_SelfAttention_Cin/{}_0/'.format(env_name)
A2C_DualAttention_log_dir = 'attention_exp/A2C_DualAttention/{}_0/'.format(env_name)
A2C_SelfAttention_log_dir = 'attention_exp/A2C_SelfAttention/{}_0/'.format(env_name)
A2C_Attention4_log_dir = 'attention_exp/A2C_Attention4/{}_0/'.format(env_name)
A2C_Attention3_log_dir = 'attention_exp/A2C_Attention3/{}_0/'.format(env_name)
A2C_Attention2_log_dir = 'attention_exp/A2C_Attention2/{}_0/'.format(env_name)
A2C_Attention_log_dir = 'attention_exp/A2C_Attention/{}_0/'.format(env_name)
A2C_log_dir = 'attention_exp/A2C/{}_0/'.format(env_name)

run('A2C_SelfAttention_Cin', env_name, num_cpu, A2C_SelfAttention_Cin_log_dir)
# run('A2C_DualAttention', env_name, num_cpu, A2C_DualAttention_log_dir)
# run('A2C_SelfAttention', env_name, num_cpu, A2C_SelfAttention_log_dir)
# run('A2C_Attention4', env_name, num_cpu, A2C_Attention4_log_dir)
# run('A2C_Attention3', env_name, num_cpu, A2C_Attention3_log_dir)
# run('A2C_Attention2', env_name, num_cpu, A2C_Attention2_log_dir)
# run('A2C_Attention', env_name, num_cpu, A2C_Attention_log_dir)
# run('A2C', env_name, num_cpu, A2C_log_dir)
print('finish')

# test('A2C_SelfAttention', env_name, num_cpu, A2C_SelfAttention_log_dir)
