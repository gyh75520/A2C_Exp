from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
import gym
env = gym.make('Tutankham-ram-v4')
# env = make_atari('BreakoutNoFrameskip-v4')
# env = VecFrameStack(env, n_stack=4)

# model = DQN(CnnPolicy, env, verbose=1)
# model.learn(total_timesteps=2500)
# model.save("dqn_breakout")
print(env.env.action_space)
obs = env.reset()
last_obs = 0
while True:
    # action, _states = model.predict(obs)
    action = int(input("action:"))
    obs, rewards, done, info = env.step(action)
    print('obs', obs[70], "x", obs[72], obs[88])
    print(obs - last_obs)
    last_obs = obs
    env.render()
