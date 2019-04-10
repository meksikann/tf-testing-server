from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym
import random


UP_ACTION = 2
DOWN_ACTION = 3

def start_pong():
    env = gym.make('Pong-v0')

    observation = env.reset()

    for i in range(3000):
        #render frame
        env.render()

        action = random.randint(UP_ACTION, DOWN_ACTION)

        # run next step
        observation, reward, done, info = env.step(action)

        # if episode is over - reset the env
        if done:
            print('reset env')
            env.reset()



start_pong()
#
# def start_atary():
#
#     env = gym.make('BreakoutDeterministic-v4')
#
#     frame = env.reset()
#
#     env.render()
#
#     is_done = False
#
#     while not is_done:
#         frame, reward, is_done, _ = env.step(env.action_space.sample())
#         env.render()
#
#     print ('Start reinforcement learning testing')
#
#
# def to_gray(img):
#     return np.mean(img, axis=2).astype(np.uint8)
#
#
# def downsample(img):
#     return img[::2, ::2]
#
#
# def prepprocess(img):
#     return to_gray(downsample(img))
#
#
# def transform_reward(reward):
#     return np.sign(reward)
#
#
# def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
#     """one DQL iteration
#
#     :param model: DQN
#     :param gamma: discount factor
#     :param start_states: np.array
#     :param actions: np.array with one-hot encoded actions regarding start_states
#     :param rewards: np.array regarding to start_states received
#     :param next_states: np,.array of
#     """
#     next_Q_values = model.predict([next_states, np.ones(actions.shape)])
#
#     next_Q_values[is_terminal] = 0
#
#     start_Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
#
#     model.fit(
#         [start_states, actions], actions * start_Q_values[:, None],
#         nb_epoch=1, batch_size=len(start_states), verbose=0
#     )
#
#
#
# def q_iteration(env, model, state, iteration, memory):
#     # Choose epsilon based on the iteration
#     epsilon = get_epsilon_for_iteration(iteration)
#
#     # Choose the action
#     if random.random() < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = choose_best_action(model, state)
#
#     # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
#     new_frame, reward, is_done, _ = env.step(action)
#     memory.add(state, action, new_frame, reward, is_done)
#
#     # Sample and fit
#     batch = memory.sample_batch(32)
#     fit_batch(model, batch)
#
#
#
#
