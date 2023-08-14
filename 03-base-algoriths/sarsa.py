import numpy as np

import matplotlib.pyplot as plt

import gymnasium as gym

import warnings

from utils import render_video

warnings.filterwarnings('ignore')


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon

    return np.random.choice(np.arange(action_n), p=policy)


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)  # Создаем массив для хранения общих вознаграждений для каждого эпизода
    
    state_n = env.observation_space.n  # Получаем количество состояний в среде
    action_n = env.action_space.n  # Получаем количество действий в среде
    qfunction = np.zeros((state_n, action_n))  # Создаем Q-функцию (матрицу состояние-действие) и инициализируем её нулями
    
    for episode in range(episode_n):  # Запускаем цикл для каждого эпизода
        epsilon = 1 / (episode + 1)  # Уменьшаем параметр epsilon для epsilon-жадной стратегии с каждым эпизодом
        
        state = env.reset()[0]  # Сбрасываем среду и получаем начальное состояние
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)  # Получаем действие с использованием epsilon-жадной стратегии
        
        for _ in range(trajectory_len):  # Запускаем цикл для каждого шага внутри эпизода (ограниченного trajectory_len)
            next_state, reward, done, _, _ = env.step(action)  # Выполняем выбранное действие и получаем следующее состояние, вознаграждение и флаг завершения
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)  # Получаем следующее действие с использованием epsilon-жадной стратегии
            
            qfunction[state][action] += alpha * (reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])  # Обновляем Q-функцию согласно формуле метода SARSA
            
            state = next_state  # Переходим в следующее состояние
            action = next_action  # Переходим в следующее действие
            
            total_rewards[episode] += reward  # Добавляем полученное вознаграждение к общему вознаграждению текущего эпизода
            
            if done:  # Если эпизод завершился, выходим из цикла
                break

    return total_rewards, qfunction  # Возвращаем массив общих вознаграждений для каждого эпизода


if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    total_rewards, qfunction = SARSA(env, episode_n=500, trajectory_len=1000, gamma=0.999, alpha=0.5)
    plt.plot(total_rewards)
    plt.savefig('monte_carlo_rewards.png')

    render_video(env, qfunction, video_title="sarsa", fps=2)
