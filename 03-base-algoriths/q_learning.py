import numpy as np

import matplotlib.pyplot as plt

import gymnasium as gym

from utils import render_video

import warnings
warnings.filterwarnings('ignore')


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon

    return np.random.choice(np.arange(action_n), p=policy)


def QLearning(env, episode_n, noisy_episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)  # Создаем массив для хранения общих вознаграждений для каждого эпизода

    state_n = env.observation_space.n  # Получаем количество состояний в среде
    action_n = env.action_space.n  # Получаем количество действий в среде
    Q = np.zeros((state_n, action_n))  # Создаем Q-функцию (матрицу состояние-действие) и инициализируем её нулями

    epsilon = 1
    for episode in range(episode_n):  # Запускаем цикл для каждого эпизода
        # epsilon = 1 / (episode + 1)  # Уменьшаем параметр epsilon для epsilon-жадной стратегии с каждым эпизодом

        if episode % 100 == 0:
            print(f'#{episode} epsilon: {epsilon}')

        state = env.reset()[0]

        for _ in range(trajectory_len):  # Запускаем цикл для каждого шага внутри эпизода (ограниченного trajectory_len)
            action = get_epsilon_greedy_action(Q[state], epsilon, action_n)  # Получаем действие с использованием epsilon-жадной стратегии

            next_state, reward, done, _, _ = env.step(action)  # Выполняем выбранное действие и получаем следующее состояние, вознаграждение и флаг завершения
            # next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)  # Получаем следующее действие с использованием epsilon-жадной стратегии

            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])  # Обновляем Q-функцию согласно формуле метода SARSA

            # state = next_state  # Переходим в следующее состояние
            # action = next_action  # Переходим в следующее действие

            total_rewards[episode] += reward  # Добавляем полученное вознаграждение к общему вознаграждению текущего эпизода

            if done:  # Если эпизод завершился, выходим из цикла
                break

            state = next_state

        epsilon = max(0, epsilon - 1 / noisy_episode_n)  # Уменьшаем epsilon с течением времени для уменьшения исследования в пользу использования текущей стратегии

    return total_rewards, Q  # Возвращаем массив общих вознаграждений для каждого эпизода


def main():
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    total_rewards, qfunction = QLearning(env, episode_n=1000, noisy_episode_n=1, trajectory_len=500, gamma=0.93, alpha=0.97)
    plt.plot(total_rewards)
    plt.savefig('q_learning.png')

    mean_reward = np.mean(total_rewards[:-100])
    print(mean_reward)
    if mean_reward > -40:
        render_video(env, qfunction, video_title="q_learning", fps=2)


if __name__ == '__main__':
    main()