import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import random
from itertools import accumulate

import torch
import torch.nn as nn
import torch.optim as optim


class MADDPG_Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(MADDPG_Actor, self).__init__()

        c, h, w = obs_size

        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=7, stride=4)
        self.relu1 = nn.ELU(alpha=0.1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=3)
        self.relu2 = nn.ELU(alpha=0.1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ELU(alpha=0.1)
        self.linear1 = nn.LazyLinear(2000)
        self.linear1_relu = nn.ELU(alpha=0.1)
        self.linear2 = nn.Linear(2000, 200)
        self.linear2_relu = nn.ELU(alpha=0.1)
        self.linear3 = nn.Linear(200, n_actions)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.relu3(x)
        # print(x.shape)
        x = x.flatten(-3, -1)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = self.linear1_relu(x)
        # print(x.shape)
        x = self.linear2(x)
        # print(x.shape)
        x = self.linear2_relu(x)
        # print(x.shape)
        x = self.linear3(x)

        return x


class MADDPG_Critic(nn.Module):
    def __init__(self, full_obs_size, n_actions_agents):
        super(MADDPG_Critic, self).__init__()

        c, h, w = full_obs_size

        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=7, stride=4)
        self.relu1 = nn.ELU(alpha=0.1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=3)
        self.relu2 = nn.ELU(alpha=0.1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ELU(alpha=0.1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = nn.LazyLinear(64)
        self.linear1_relu = nn.ELU(alpha=0.1)
        self.linear2 = nn.Linear(64, 32)
        self.linear2_relu = nn.ELU(alpha=0.1)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = self.conv1(state)
        # print(x.shape)
        x = self.relu1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.relu3(x)
        # print(x.shape)
        x = self.avg_pooling(x)
        # print(x.shape)
        x = x.squeeze((-2, -1))
        # print(x.shape)
        x = torch.cat([x, action], dim=-1)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = self.linear1_relu(x)
        # print(x.shape)
        x = self.linear2(x)
        # print(x.shape)
        x = self.linear2_relu(x)
        # print(x.shape)
        x = self.linear3(x)

        return x


# Выбираем возможное действие с максимальным из стратегии действий
# с учетом дополнительного случайного шума
def select_actionFox(act_prob, avail_actions_ind, n_actions, noise_rate):
    # p = np.random.random(1).squeeze()
    # Добавляем случайный шум к действиям для исследования
    # разных вариантов действий
    zero = True
    zero_count = 0
    while zero:
        zero_count += 1
        zero = False
        for i in range(n_actions):
            # Создаем шум заданного уровня
            noise = noise_rate * (np.random.rand())
            # Добавляем значение шума к значению вероятности выполнения действия
            act_prob[i] = act_prob[i] + noise
        if list(accumulate(act_prob))[-1] <= 0.0:
            zero = True
        if zero_count > 100:
            raise StopIteration("Cumulative sum of weights less than 0")

        # Выбираем действия в зависимости от вероятностей их выполнения
        try:
            actiontemp = random.choices(['0', '1', '2'],
                                        weights=[act_prob[0], act_prob[1], act_prob[2]])

            # Преобразуем тип данных
            action = int(actiontemp[0])
            # Проверяем наличие выбранного действия в списке действий
            if action in avail_actions_ind:
                return action
            else:
                act_prob[action] = 0
        except ValueError:
            act_prob[0] = 0
            act_prob[1] = 0
            act_prob[2] = 0


# Создаем минивыборку определенного объема из буфера воспроизведения
def sample_from_expbuf(experience_buffer, batch_size):
    # Функция возвращает случайную последовательность заданной длины
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    # Минивыборка
    exp_obs=[]
    exp_acts = []
    exp_next_obs = []
    exp_next_acts = []
    exp_rew = []
    exp_termd = []

    for i in range(len(experience_buffer)):
        if i in perm_batch:
            exp_obs.append(experience_buffer[i][0])
            exp_acts.append(experience_buffer[i][1])
            exp_next_obs.append(experience_buffer[i][2])
            exp_next_acts.append(experience_buffer[i][3])
            exp_rew.append(experience_buffer[i][4])
            exp_termd.append(experience_buffer[i][5])

    # experience = np.array(experience_buffer)[perm_batch]
    # Возвращаем значения минивыборки по частям
    return (np.array(exp_obs), np.array(exp_acts),
            np.array(exp_next_obs), np.array(exp_next_acts),
            np.array(exp_rew), np.array(exp_termd))
    # return experience[:, 0], experience[:, 1], experience[:, 2], experience[:, 3], experience[:, 4], experience[:, 5]

