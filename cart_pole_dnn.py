import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def prepare_training_data_from_random_play(env, games, max_steps):
    training_data = []
    for _ in range(games):
        env.reset()
        total_reward = 0
        game_memory = []
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)
        for _ in range(max_steps):
            action = random.randrange(0, 2)
            game_memory.append([observation, action])
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        if total_reward >= 60:
            for observation_and_action in game_memory:
                if observation_and_action[1] == 1:
                    training_data.append([observation_and_action[0], [0, 1]])
                else:
                    training_data.append([observation_and_action[0], [1, 0]])
    return training_data


def build_and_train_model(observations, actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_dim=4, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    history = model.fit(x=observations, y=actions, epochs=10)
    plot_training_stats(history)
    return model


def calculate_next_action(model, observation):
    obs_reshaped = observation.reshape(-1, len(observation))
    return np.argmax(model.predict(obs_reshaped)[0])


# def build_and_train_model(observations, actions):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(24, input_dim=4, activation='relu'),
#         tf.keras.layers.Dense(24, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(loss='mse', optimizer='adam', metrics=['acc'])
#     history = model.fit(x=observations, y=actions, epochs=10)
#     plot_training_stats(history)
#     return model
#
#
# def calculate_next_action(model, observation):
#     obs_reshaped = observation.reshape(-1, len(observation))
#     action_prob = model.predict(obs_reshaped)[0][0]
#     return int(round(action_prob))


def play_game(env, model, times):
    print('Playing {} games...'.format(times))
    game = 0
    for _ in range(times):
        env.reset()
        action = 0
        score = 0
        for _ in range(1000):
            observation, reward, done, info = env.step(action)
            # env.render()
            score += reward
            if done:
                print('Game {} score: {}'.format(game, score))
                game += 1
                break
            action = calculate_next_action(model, observation)


def plot_training_stats(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


env = gym.make('CartPole-v1')
training_data = prepare_training_data_from_random_play(env, 10000, 500)
model = build_and_train_model(
    np.array([i[0] for i in training_data]),
    np.array([i[1] for i in training_data]))
play_game(env, model, 10)
