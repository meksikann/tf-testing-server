from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym


env = gym.make('BreakoutDeterministic-v4')

frame = env.reset()

env.render()

is_done = False

while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    env.render()

print ('Start reinforcement learning testing')
