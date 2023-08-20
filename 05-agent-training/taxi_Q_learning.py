import random
import matplotlib.pyplot as plt
import time

import numpy as np
import gymnasium as gym

from utils import render_video


class QLearning:
    def __init__(self, state_n, action_n, noisy_episode_n=200, alpha=0.5, gamma=0.99):
        self.state_n = state_n
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.noisy_episode_n = noisy_episode_n
        # Инициализация q-функции нулями
        self.qfunction = np.zeros((self.state_n, self.action_n))

    def get_epsilon_action(self, state, epsilon):
        # epsilon = max(0, self.epsilon - 1 / self.noisy_episode_n)
        # print(epsilon)

        # Выбор действия на основе эпсилон-жадной стратегии
        if random.random() < epsilon:
            return random.randint(0, self.action_n - 1)
        else:
            return np.argmax(self.qfunction[state])

    def update_qfunction(self, state, action, reward, next_state):
        # Обновление q-функции с использованием алгоритма Q-learning
        current_q = self.qfunction[state][action]
        next_q = np.max(self.qfunction[next_state])
        updated_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.qfunction[state][action] = updated_q


def main():
    experiment_name = 'taxi_q_learning'

    env = gym.make('Taxi-v3')

    state_n = env.observation_space.n
    agent = QLearning(state_n, env.action_space.n, alpha=0.2, gamma=0.9)
    episode_n = 2000
    trajectory_len = 500

    epsilon = 1
    episodes_reward = []
    for episode in range(episode_n):
        state = env.reset()[0]

        epsilon = 0
        if episode < 200:
            epsilon = 1 / (episode + 1)
            print(f'#{episode} epsilon: {epsilon}')

        episode_reward = 0
        for _ in range(trajectory_len):
            # Выбор действия на основе текущего состояния
            action = agent.get_epsilon_action(state, epsilon)

            # Выполнение выбранного действия и получение нового состояния и вознаграждения
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # Обновление q-функции по алгоритму Q-learning
            agent.update_qfunction(state, action, reward, next_state)

            # Переход к следующему состоянию
            state = next_state

            if done:
                break

        print(episode, episode_reward)
        episodes_reward.append(episode_reward)

    plt.plot(episodes_reward)
    plt.savefig('%s.png' % experiment_name)

    episodes_reward = episodes_reward[-100:]
    print('Min reward:', np.min(episodes_reward))
    print('Max reward:', np.max(episodes_reward))
    print('Average reward:', np.mean(episodes_reward))
    print('STD reward:', np.std(episodes_reward))

    env = gym.make("Taxi-v3", render_mode="rgb_array")
    render_video(env, agent.qfunction, video_title=experiment_name, fps=2)

    # env = gym.make('Taxi-v3', render_mode='human')
    # state = env.reset()[0]
    #
    # for _ in range(trajectory_len):
    #     action = agent.get_epsilon_action(state, 0)
    #
    #     # Выполнение действия в окружении и получение нового состояния
    #     state, _, done, _, _ = env.step(action)
    #
    #     env.render()  # Визуализация окружения
    #     time.sleep(0.2)
    #
    #     if done:
    #         break


if __name__ == '__main__':
    main()
