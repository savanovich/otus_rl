from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.butterfly import cooperative_pong_v5

from maddpg import MADDPG_Actor, MADDPG_Critic, select_actionFox, sample_from_expbuf


if __name__ == '__main__':
    env = cooperative_pong_v5.env(render_mode='rgb_array') #render_mode="human")
    env.reset(seed=42)

    print(env.num_agents)
    print(env.max_num_agents)

    n_agents = env.num_agents
    AGENT_IDX_MAP = {
        'paddle_0': 0,
        'paddle_1': 1,
    }

    # print(env.action_space('paddle_0'))
    # print(env.action_space('paddle_1'))

    n_actions = env.action_space('paddle_1')
    n_actions = n_actions.n

    # print(env.observation_space('paddle_0'))
    # print(env.observation_space('paddle_1'))

    observation_space = env.observation_space('paddle_1')
    print(observation_space.shape)
    obs_size = observation_space.shape

    print("n_actions=", n_actions)
    print("obs_size=", obs_size)

    obs_size = obs_size[2], obs_size[0], obs_size[1]

    # Определяем основные параметры нейросетевого обучения
    ##########################################################################
    # Некоторые переходы в алгоритме MADDPG зависят от шагов игры
    global_step = 0  # подсчитываем общее количество шагов в игре
    start_steps = 1000  # начинаем обучать через 1000 шагов
    # start_steps = 100  # начинаем обучать через 1000 шагов
    steps_train = 4  # после начала обучения продолжаем обучать каждый 4 шаг
    # Размер минивыборки
    batch_size = 64
    # Общее количество эпизодов игры
    # n_episodes = 2
    n_episodes = 500
    # Параметр дисконтирования.
    gamma = 0.99
    # Скорость обучения исполнителя
    alpha_actor = 0.001
    # Скорость обучения критика
    alpha_critic = 0.001

    # Уровень случайного шума
    noise_rate = 0.01
    # Начальное значение случайного шума
    noise_rate_max = 0.9
    # Финальное значение случайного шума
    noise_rate_min = 0.01
    # Шаг затухания уровня случайного шума
    # noise_decay_steps = 15000
    noise_decay_steps = 2000

    # Параметр мягкой замены
    tau = 0.01
    # Объем буфера воспроизведения
    # buffer_len = 10000
    buffer_len = 3000
    ###########################################################################

    # Создаем буфер воспроизведения на основе deque
    experience_buffer = deque(maxlen=buffer_len)

    # Pytorch определяет возможность использования графического процессора
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print('DEVICE:', device)

    # Реализуем модифицированный алгоритм MADDPG
    # с одной нейронной сетью критика и тремя нейронными сетями исполнителей

    # Создаем основную нейронную сеть критика
    critic_network = MADDPG_Critic([obs_size[0] * n_agents, obs_size[1], obs_size[2]], n_actions).to(device)
    # Создаем целевую нейронную сеть критика
    tgtCritic_network = MADDPG_Critic([obs_size[0] * n_agents, obs_size[1], obs_size[2]], n_actions).to(device)
    tgtCritic_network.load_state_dict(critic_network.state_dict())


    # Создаем списки для мультиагентного случая
    actor_network_list = []
    tgtActor_network_list = []
    optimizerActor_list = []
    # objectiveActor_list = []

    for agent_id in range(n_agents):
        # Создаем основную нейронную сеть исполнителя
        actor_network = MADDPG_Actor(obs_size, n_actions).to(device)
        # Создаем целевую нейронную сеть исполнителя
        tgtActor_network = MADDPG_Actor(obs_size, n_actions).to(device)
        tgtActor_network.load_state_dict(actor_network.state_dict())

        # Создаем список основных нейронных сетей исполнителей для трех агентов
        actor_network_list.append(actor_network)
        # Создаем список целевых нейронных сетей исполнителей
        tgtActor_network_list.append(tgtActor_network)
        # Создаем список оптимизаторов нейронных сетей исполнителей
        optimizerActor_list.append(optim.AdamW(params=actor_network_list[agent_id].parameters(), lr=alpha_actor))
        # Создаем список функций потерь исполнителей
        # objectiveActor_list.append(nn.MSELoss())

    # Создаем оптимизатор нейронной сети критика
    optimizerCritic = optim.AdamW(params=critic_network.parameters(), lr=alpha_critic)
    # Создаем функцию потерь критика
    objectiveCritic = nn.MSELoss()

    # Выводим на печать архитектуру нейронных сетей
    # print('Actor_network_list=', actor_network_list)
    # print('Critic_network_list=', critic_network)

    # Определяем вспомогательные параметры
    Loss_History = []
    Loss_History_actor = []

    Reward_History = []
    winrate_history = []

    total_loss = []
    total_loss_actor = []

    m_loss = []
    m_loss_actor = []

    max_reward = -1000

    for e in range(n_episodes):
        print(f'=============EPISODE #{e} ============')

        env.reset()
        terminated = False
        episode_reward = 0
        # Обновляем и выводим динамический уровень случайного шума
        noise_rate = max(noise_rate_min,
                         noise_rate_max - (noise_rate_max - noise_rate_min) * global_step / noise_decay_steps)
        print('NOISE RATE:', noise_rate)

        # Шаги игры внутри эпизода
        while not terminated:
            # Обнуляем промежуточные переменные
            actions = []
            observations = []
            action = 0
            # Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents])
            # Храним историю состояний среды один шаг для разных агентов
            obs_agent = np.zeros([n_agents, obs_size[0], obs_size[1], obs_size[2]//2], dtype=object)         # TODO: !!!!!!!!!
            obs_agent_next = np.zeros([n_agents, obs_size[0], obs_size[1], obs_size[2]//2], dtype=object)

            ###########_Цикл по агентам для выполнения действий в игре_########
            for agent_id in range(n_agents):
                observation, reward, terminated, truncation, info = env.last()

                if agent_id == 0:
                    observation = observation[:,:observation.shape[1] // 2,:]
                else:
                    observation = observation[:,observation.shape[1] // 2:,:]

                # import cv2
                # cv2.imwrite(f'obs_{agent_id}.png', observation)

                observation = np.transpose(observation, (2, 0, 1))

                obs_agent[agent_id] = observation
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
                # Передаем состояние среды в основную нейронную сеть
                # и получаем стратегию действий
                action_probabilitiesT = actor_network_list[agent_id](obs_agentT)
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                # print('action_probabilities', action_probabilities)

                # Находим возможные действия агента в данный момент времени
                avail_actions = np.ones(n_actions)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # Выбираем возможное действие агента с учетом
                # стратегии действий и уровня случайного шума
                action = select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)
                # Обрабатываем исключение при ошибке в возможных действиях
                if action is None:
                    action = np.random.choice(avail_actions_ind)

                # Собираем действия от разных агентов
                actions.append(action)
                actionsFox[agent_id] = action
                # Собираем локальные состояния среды от разных агентов
                observations.append(obs_agent[agent_id])

                if terminated or truncation:
                    env.step(None)
                else:
                    env.step(action)

                episode_reward += reward

            # Подготовляем данные для сохранения в буфере воспроизведения
            actions_next = []
            observations_next = []
            # Если эпизод не завершился, то можно найти новые действия и состояния
            if not terminated:
                for agent_id in range(n_agents):
                    # Получаем новое состояние среды для независимого агента
                    observation, reward, terminated, truncation, info = env.last()

                    if agent_id == 0:
                        observation = observation[:, :observation.shape[1] // 2, :]
                    else:
                        observation = observation[:, observation.shape[1] // 2:, :]

                    observation = np.transpose(observation, (2, 0, 1))
                    obs_agent_next[agent_id] =observation
                    # Собираем от разных агентов новые состояния
                    observations_next.append(obs_agent_next[agent_id])
                    obs_agent_nextT = torch.FloatTensor([obs_agent_next[agent_id]]).to(device)
                    # Получаем новые действия агентов для новых состояний
                    # из целевой сети исполнителя
                    action_probabilitiesT = tgtActor_network_list[agent_id](obs_agent_nextT)
                    # Конвертируем данные в numpy
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    # Находим новые возможные действия агента
                    avail_actions = np.ones(n_actions)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    # Выбираем новые возможные действия
                    action = select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)
                    if action is None:
                        action = np.random.choice(avail_actions_ind)
                    actions_next.append(action)

                    if terminated or truncation:
                        env.step(None)
                    else:
                        env.step(action)

            elif terminated:
                # если эпизод на этом шаге завершился, то новых действий не будет
                actions_next = actions
                observations_next = observations

            # Сохраняем переход в буфере воспроизведения
            experience_buffer.append([observations, actions, observations_next, actions_next, reward, terminated])
            # print(len(experience_buffer))

            # Если буфер воспроизведения наполнен, начинаем обучать сеть
            if (global_step % steps_train == 0) and (global_step > start_steps):
                # Получаем минивыборку из буфера воспроизведения
                exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = sample_from_expbuf(
                    experience_buffer, batch_size)

                # Конвертируем данные в тензор
                # exp_obs = [x for x in exp_obs]
                exp_obs = [np.concatenate([x[0], x[1]], axis=-1) for x in exp_obs]
                obs_agentsT = torch.FloatTensor(exp_obs).to(device)
                exp_acts = [x for x in exp_acts]
                act_agentsT = torch.FloatTensor(exp_acts).to(device)

                ###############_Обучаем нейронную сеть критика_################

                # Получаем значения из основной сети критика
                action_probabilitiesQT = critic_network(obs_agentsT, act_agentsT)
                action_probabilitiesQT = action_probabilitiesQT.to("cpu")

                # Конвертируем данные в тензор
                exp_next_obs = [x for x in exp_next_obs]
                exp_next_obs = [np.concatenate([x[0], x[1]], axis=-1) for x in exp_next_obs]
                obs_agents_nextT = torch.FloatTensor(exp_next_obs).to(device)
                exp_next_acts = [x for x in exp_next_acts]
                act_agents_nextT = torch.FloatTensor(exp_next_acts).to(device)

                # Получаем значения из целевой сети критика
                action_probabilitiesQ_nextT = tgtCritic_network(obs_agents_nextT, act_agents_nextT)
                action_probabilitiesQ_nextT = action_probabilitiesQ_nextT.to("cpu")
                action_probabilitiesQ_next = action_probabilitiesQ_nextT.data.numpy()

                # Переформатируем y_batch размером batch_size
                y_batch = np.zeros([batch_size])
                action_probabilitiesQBT = torch.empty(batch_size, dtype=torch.float)

                for i in range(batch_size):
                    # Вычисляем целевое значение y
                    y_batch[i] = exp_rew[i] + (gamma * action_probabilitiesQ_next[i][0]) * (1 - exp_termd[i])
                    action_probabilitiesQBT[i] = action_probabilitiesQT[i][0]

                y_batchT = torch.FloatTensor(y_batch)

                # Обнуляем градиенты
                optimizerCritic.zero_grad()

                # Вычисляем функцию потерь критика
                loss_t_critic = objectiveCritic(action_probabilitiesQBT, y_batchT)

                # Сохраняем данные для графиков
                Loss_History.append(loss_t_critic)
                loss_n_critic = loss_t_critic.data.numpy()
                total_loss.append(loss_n_critic)
                critic_loss_mean = np.mean(total_loss[-1000:])
                m_loss.append(critic_loss_mean)
                print('Critic Loss', critic_loss_mean)

                # Выполняем обратное распространение ошибки для критика
                loss_t_critic.backward()

                # Выполняем оптимизацию нейронной сети критика
                optimizerCritic.step()
                ###################_Закончили обучать критика_#################

                ##############_Обучаем нейронные сети исполнителей_############
                # Разбираем совместное состояние на локальные состояния
                obs_local1 = np.zeros([batch_size, obs_size[0], obs_size[1], obs_size[2]//2])
                obs_local2 = np.zeros([batch_size, obs_size[0], obs_size[1], obs_size[2]//2])
                for i in range(batch_size):
                    obs_local1[i] = exp_obs[i][:,:,:exp_obs[i].shape[2]//2]
                    obs_local2[i] = exp_obs[i][:,:,exp_obs[i].shape[2]//2:]
                # for i in range(batch_size):
                # for i in range(batch_size):
                #     k = 0
                #     for j in range(obs_size, obs_size * 2):
                #         obs_local2[i][k] = exp_obs[i][j]
                #         k = k + 1
                # for i in range(batch_size):
                #     k = 0
                #     for j in range(obs_size * 2, obs_size * 3):
                #         obs_local3[i][k] = exp_obs[i][j]
                #         k = k + 1
                # Конвертируем данные в тензор
                obs_agentT1 = torch.FloatTensor(obs_local1).to(device)
                obs_agentT2 = torch.FloatTensor(obs_local2).to(device)
                # obs_agentT3 = torch.FloatTensor([obs_local3]).to(device)

                # Обнуляем градиенты
                optimizerActor_list[0].zero_grad()
                optimizerActor_list[1].zero_grad()

                # Подаем в нейронные сети исполнителей локальные состояния
                action_probabilitiesT1 = actor_network_list[0](obs_agentT1)
                action_probabilitiesT2 = actor_network_list[1](obs_agentT2)

                # Конвертируем данные в numpy
                action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
                action_probabilitiesT2 = action_probabilitiesT2.to("cpu")
                # action_probabilitiesT3 = action_probabilitiesT3.to("cpu")
                action_probabilities1 = action_probabilitiesT1.data.numpy()
                action_probabilities2 = action_probabilitiesT2.data.numpy()
                # action_probabilities3 = action_probabilitiesT3.data.numpy()[0]

                # Вычисляем максимальные значения с учетом объема минивыборки
                act_full = np.zeros([batch_size, n_agents])
                for i in range(batch_size):
                    act_full[i][0] = np.argmax(action_probabilities1[i])
                    act_full[i][1] = np.argmax(action_probabilities2[i])
                act_fullT = torch.FloatTensor(act_full).to(device)

                # Конвертируем данные в тензор
                # exp_obs = [x for x in exp_obs]
                obs_agentsT = torch.FloatTensor(exp_obs).to(device)

                # Задаем значение функции потерь для нерйонных сетей исполнителей
                # как отрицательный выход критика
                actor_lossT = -critic_network(obs_agentsT, act_fullT)

                # Усредняем значение по количеству элементов минивыборки
                actor_lossT = actor_lossT.mean()

                # Выполняем обратное распространение ошибки
                actor_lossT.backward()

                # Выполняем оптимизацию нейронных сетей исполнителей
                optimizerActor_list[0].step()
                optimizerActor_list[1].step()

                # Собираем данные для графиков
                actor_lossT = actor_lossT.to("cpu")
                Loss_History_actor.append(actor_lossT)
                actor_lossN = actor_lossT.data.numpy()
                total_loss_actor.append(actor_lossN)
                m_loss_actor.append(np.mean(total_loss_actor[-1000:]))
                print('Actor Loss:', np.mean(total_loss_actor[-1000:]))
                ##############_Закончили обучать исполнителей_#################

                # Рализуем механизм мягкой замены
                # Обновляем целевую сеть критика
                for target_param, param in zip(tgtCritic_network.parameters(), critic_network.parameters()):
                    target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
                # Обновляем целевые сети акторов
                for agent_id in range(n_agents):
                    for target_param, param in zip(tgtActor_network_list[agent_id].parameters(),
                                                   actor_network_list[agent_id].parameters()):
                        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

                ######################_конец if обучения_######################

            # Обновляем счетчик общего количества шагов
            global_step += 1

        ######################_конец цикла while_##############################

        # Выводим на печать счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        # Собираем данные для графиков
        Reward_History.append(episode_reward)
        # status = env.get_stats()
        # winrate_history.append(status["win_rate"])
        winrate_history.append(global_step > 1000)

    ################_конец цикла по эпизодам игры_#############################

        if len(m_loss):
            plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            plt.plot(Reward_History)
            plt.xlabel('Номер эпизода')
            plt.ylabel('Количество награды за эпизод')
            plt.savefig('plots/reward.png')
            plt.close()

            # plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            # plt.plot(winrate_history)
            # plt.xlabel('Номер эпизода')
            # plt.ylabel('Процент побед')
            # plt.savefig('reward.png')
            # plt.close()

            plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            plt.plot(m_loss_actor)
            plt.xlabel('Номер каждой 1000 итерации')
            plt.ylabel('Функция потерь исполнителя')
            plt.savefig('plots/actor_loss.png')
            plt.close()

            # Значения функции потерь критика
            plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            plt.plot(m_loss)
            plt.xlabel('Номер каждой 1000 итерации')
            plt.ylabel('Функция потерь критика')
            plt.savefig('plots/critic_loss.png')
            plt.close()

            plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            plt.plot(np.sum(np.stack([np.array(m_loss), np.array(m_loss_actor)]), axis=0).tolist())
            plt.xlabel('Номер каждой 1000 итерации')
            plt.ylabel('Функция потерь общая')
            plt.savefig('plots/total_loss.png')
            plt.close()

            if episode_reward > max_reward and global_step > start_steps:
                print('BEST EPISODE')
                max_reward = episode_reward
                for agent_id in range(n_agents):
                    torch.save(actor_network_list[agent_id].state_dict(), "output/actornet_%.0f_%s_BEST.dat" % (agent_id, e))
            else:
                for agent_id in range(n_agents):
                    torch.save(actor_network_list[agent_id].state_dict(), "output/actornet_%.0f_%s.dat" % (agent_id, e))

    env.close()
