from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os
import numpy as np





log_dir = 'test/dqn_breakout/'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

env = make_atari('BreakoutNoFrameskip-v4')

model= DQN.load("test/dqn_breakout/best_model")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
