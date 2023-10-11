from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.butterfly import cooperative_pong_v5

from maddpg import MADDPG_Actor, MADDPG_Critic, select_actionFox, sample_from_expbuf


if __name__ == '__main__':
    env = cooperative_pong_v5.env(render_mode="rgb_array")
    env.reset(seed=42)

    print(env.num_agents)
    print(env.max_num_agents)

    n_agents = env.num_agents
    AGENT_IDX_MAP = {
        'paddle_0': 0,
        'paddle_1': 1,
    }

    print(env.action_space('paddle_0'))
    print(env.action_space('paddle_1'))

    n_actions = env.action_space('paddle_1')
    n_actions = n_actions.n

    print(env.observation_space('paddle_0'))
    print(env.observation_space('paddle_1'))

    observation_space = env.observation_space('paddle_1')
    print(observation_space.shape)
    obs_size = observation_space.shape

    print("n_actions=", n_actions)
    print("obs_size=", obs_size)

    obs_size = obs_size[2], obs_size[0], obs_size[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)

    actor_1 = MADDPG_Actor(obs_size, n_actions).to(device)
    checkpoint_1 = torch.load("output/actornet_0_354.dat")
    actor_1.load_state_dict(checkpoint_1)
    actor_2 = MADDPG_Actor(obs_size, n_actions).to(device)
    checkpoint_2 = torch.load("output/actornet_1_354.dat")
    actor_2.load_state_dict(checkpoint_2)
    actor_network_list = [actor_1, actor_2]

    for e in range(1):
        print(f'=============EPISODE #{e} ============')

        env.reset()
        terminated = False
        episode_reward = 0
        image_obs = []
        while not terminated:
            action = 0
            for agent_id in range(n_agents):
                observation, reward, terminated, truncation, info = env.last()
                image_obs.append(observation)

                if agent_id == 0:
                    observation = observation[:, :observation.shape[1] // 2, :]
                else:
                    observation = observation[:, observation.shape[1] // 2:, :]

                observation = np.transpose(observation, (2,0,1))

                obs_agentT = torch.FloatTensor([observation]).to(device)
                action_probabilitiesT = actor_network_list[agent_id](obs_agentT)
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                action = np.argmax(action_probabilities)

                if terminated or truncation:
                    env.step(None)
                    print('TRANCATION')
                    env.reset()

                    import moviepy.editor as mpy
                    import os

                    clip = mpy.ImageSequenceClip(list(image_obs), fps=10)
                    # txt_clip = (mpy.TextClip(video_title, fontsize=30,color='white')
                    # .set_position('top', 'center')
                    # .set_duration(10))
                    video = mpy.CompositeVideoClip([clip,
                                                    # txt_clip
                                                    ])
                    new_video_title = f'video_{e}.mp4'
                    filename = os.path.join('./', new_video_title)
                    video.write_videofile(filename)

                    break
                else:
                    env.step(action)

                episode_reward += reward

        print('REWARD:', episode_reward)

    env.close()
