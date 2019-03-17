Go to the `env/gym-box-world` folder and run the command :
```
pip install -e .
```

This will install the box-world environment.Now, we can use this enviroment with the following:
```
import gym
import gym_boxworld
env = gym.make('BoxWorldNoFrameskip-v4')
```
