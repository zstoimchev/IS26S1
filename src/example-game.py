import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.astronaut import Keyboard

env = gym.make("famnit_gym/Astronaut-v1", render_mode="human")
env = Keyboard(env)

observation, info = env.reset()

try:
    while True:
        observation, reward, terminated, truncated, info = env.step(0)

        if terminated or truncated:
            break
finally:
    env.close()
