import numpy as np

import matplotlib.pyplot as plt

import gymnasium as gym

from utils import plot_heatmap, render_video

import warnings
warnings.filterwarnings('ignore')


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon

    return np.random.choice(np.arange(action_n), p=policy)


def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    total_rewards = []  # Создаем список для хранения общих вознаграждений для каждого эпизода

    state_n = env.observation_space.n  # Получаем количество состояний в среде
    action_n = env.action_space.n  # Получаем количество действий в среде
    qfunction = np.zeros((state_n, action_n))  # Создаем Q-функцию (матрицу состояние-действие) и инициализируем её нулями
    counter = np.zeros((state_n, action_n))  # Создаем счетчик для подсчета количества визитов в каждую ячейку Q-функции

    for episode in range(episode_n):  # Запускаем цикл для каждого эпизода
        epsilon = 1 - episode / episode_n  # Уменьшаем параметр epsilon для epsilon-жадной стратегии с каждым эпизодом
        trajectory = {'states': [], 'actions': [], 'rewards': []}  # Создаем структуру данных для хранения траектории эпизода

        state = env.reset()[0]  # Сбрасываем среду и получаем начальное состояние
        for trajectory_i in range(trajectory_len):  # Запускаем цикл для каждого шага внутри эпизода (ограниченного trajectory_len)
            trajectory['states'].append(state)  # Добавляем текущее состояние в траекторию

            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)  # Получаем действие с использованием epsilon-жадной стратегии
            trajectory['actions'].append(action)  # Добавляем текущее действие в траекторию

            state, reward, done, _, _ = env.step(action)  # Выполняем выбранное действие и получаем следующее состояние, вознаграждение и флаг завершения
            trajectory['rewards'].append(reward)  # Добавляем полученное вознаграждение в траекторию

            if done:  # Если эпизод завершился, выходим из цикла
                break

        total_reward = sum(trajectory['rewards'])
        total_rewards.append(total_reward)  # Добавляем суммарное вознаграждение текущего эпизода в список
        # print(total_reward)

        real_trajectory_len = len(trajectory['rewards'])  # Определяем реальную длину траектории (может быть меньше trajectory_len, если эпизод завершился раньше)
        returns = np.zeros(real_trajectory_len + 1)  # Создаем массив для хранения возвращений (returns) на каждом шаге траектории

        for t in range(real_trajectory_len - 1, -1, -1):  # Запускаем цикл для вычисления возвращений для каждого шага в траектории (в обратном порядке)
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]  # Вычисляем возвращение с учетом дисконтирования (gamma)

        for t in range(real_trajectory_len):  # Запускаем цикл для обновления Q-функции на каждом шаге в траектории
            state = trajectory['states'][t]  # Получаем текущее состояние
            action = trajectory['actions'][t]  # Получаем текущее действие
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])  # Обновляем Q-функцию согласно формуле метода Монте-Карло
            counter[state][action] += 1  # Увеличиваем счетчик визитов для соответствующей ячейки Q-функции

        # if episode % 100 == 0:
        #     plot_heatmap(qfunction, list(range(state_n)), list(range(action_n)), f'monte_carlo_{episode}')

    return total_rewards, qfunction  # Возвращаем список общих вознаграждений для каждого эпизода


if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode='rgb_array')

    total_rewards, qfunction = MonteCarlo(env, episode_n=10000, trajectory_len=1000, gamma=0.99)
    plt.plot(total_rewards)
    plt.savefig('monte_carlo_rewards.png')

    print(np.mean(total_rewards[:-100]))
    render_video(env, qfunction, video_title="monte_carlo", fps=2)
