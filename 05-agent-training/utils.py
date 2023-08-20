import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
