from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN

env = make_atari('BreakoutNoFrameskip-v4')
# env = VecFrameStack(env, n_stack=4)

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("dqn_breakout")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

'''
from stable_baselines.common.cmd_util import make_atari_env
# from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import ACER

env = make_atari_env('BreakoutNoFrameskip-v4', num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)

model = ACER(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("dqn_breakout")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
'''
