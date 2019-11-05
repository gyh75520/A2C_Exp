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
# import gym_boxworld
import gfootball.env as football_env
import tensorflow as tf

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


def make_football_env(env_name, rank, log_dir, useMonitor=True, seed=0):
    flags = tf.app.flags
    FLAGS = tf.app.flags.FLAGS

    flags.DEFINE_string('level', 'academy_empty_goal_close',
                        'Defines type of problem being solved')
    flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                     'extracted_stacked'],
                      'Observation to be used for training.')
    flags.DEFINE_enum('reward_experiment', 'scoring',
                      ['scoring', 'scoring,checkpoints'],
                      'Reward to be used for training.')
    flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                        'gfootball_impala_cnn'],
                      'Policy architecture')
    flags.DEFINE_integer('num_timesteps', int(2e6),
                         'Number of timesteps to run for.')
    flags.DEFINE_integer('num_envs', 8,
                         'Number of environments to run in parallel.')
    flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; '
                         'batch size is nsteps * nenv')
    flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
    flags.DEFINE_integer('nminibatches', 8,
                         'Number of minibatches to split one epoch to.')
    flags.DEFINE_integer('save_interval', 100,
                         'How frequently checkpoints are saved.')
    flags.DEFINE_integer('seed', 0, 'Random seed.')
    flags.DEFINE_float('lr', 0.00008, 'Learning rate')
    flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
    flags.DEFINE_float('gamma', 0.993, 'Discount factor')
    flags.DEFINE_float('cliprange', 0.27, 'Clip range')
    flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
    flags.DEFINE_bool('dump_full_episodes', False,
                      'If True, trace is dumped after every episode.')
    flags.DEFINE_bool('dump_scores', False,
                      'If True, sampled traces after scoring are dumped.')
    flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')

    def _init():
        # env = make_atari(env_id)
        env = football_env.create_environment(
            env_name=env_name, stacked=('stacked' in FLAGS.state),
            # rewards=FLAGS.reward_experiment,
            logdir=log_dir,
            enable_goal_videos=FLAGS.dump_scores and (seed == 0),
            enable_full_episode_videos=FLAGS.dump_full_episodes and (seed == 0),
            render=FLAGS.render and (seed == 0),
            dump_frequency=50 if FLAGS.render and seed == 0 else 0)
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

    # env_id = env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_name, i, log_dir) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)
    model = get_model(model_name, env, log_dir)

    total_timesteps = int(1e5)
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

        # env.render()
        plt.pause(0.000001)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
env_name = 'academy_empty_goal_close'
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
