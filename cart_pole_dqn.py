import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

BATCH_SIZE = 20


class DQNSolver:
    def __init__(self):
        self.exploration_rate = 1.0
        self.memory = deque(maxlen=1000000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=4, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(lr=0.001))

    def remember(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    def act(self, observation):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(2)
        q_values = self.model.predict(observation)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for observation, action, reward, next_observation, done in batch:
            q = reward
            if not done:
                q = (reward + 0.95 * np.amax(self.model.predict(next_observation)[0]))
            q_values = self.model.predict(observation)
            q_values[0][action] = q
            self.model.fit(observation, q_values, verbose=0)
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)


dqn_solver = DQNSolver()
env = gym.make("CartPole-v1")
run = 0
while True:
    run += 1
    observation = env.reset()
    observation = np.reshape(observation, [1, 4])
    step = 0
    while True:
        step += 1
        # env.render()
        action = dqn_solver.act(observation)
        next_observation, reward, done, info = env.step(action)
        reward = reward if not done else -reward
        next_observation = np.reshape(next_observation, [1, 4])
        dqn_solver.remember(observation, action, reward, next_observation, done)
        observation = next_observation
        if done:
            print(
                "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
            break
        dqn_solver.experience_replay()
