import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(mat, actions, states, name):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(actions)), labels=actions)
    # ax.set_yticks(np.arange(len(states)), labels=states)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(actions)):
        for j in range(len(states)):
            text = ax.text(j, i, mat[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(name)
    fig.tight_layout()
    plt.savefig(name+'.png')


def render_video(env, qfunction, video_title, fps):
    image_obs = []
    observation = env.reset()[0]
    for _ in range(1000):
        image_obs.append(env.render())

        action = np.argmax(qfunction[observation])

        observation, reward, terminated, truncated, info = env.step(action)

        image_obs.append(env.render())

        if terminated:
            break
    import moviepy.editor as mpy
    import os
    clip = mpy.ImageSequenceClip(list(image_obs), fps=fps)
    # txt_clip = (mpy.TextClip(video_title, fontsize=30,color='white')
    # .set_position('top', 'center')
    # .set_duration(10))
    video = mpy.CompositeVideoClip([clip,
                                    # txt_clip
                                    ])
    new_video_title = video_title + '.mp4'
    filename = os.path.join('./', new_video_title)
    video.write_videofile(filename)
