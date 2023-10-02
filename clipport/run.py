import numpy as np
import torch
import clip
import jax.numpy as jnp
from clipport.train import eval_step
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from IPython.display import display
import copy
'''
This file includes code derived from [saycan] by Google LLC.
Copyright 2022 Google LLC.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Source Code: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb

Modifications made on 2023-10-03:
Extracted the cliport part of the original code.
'''
def run_cliport(env,clip_model,coords, optim, obs, text, unct=None, maps=None):
  before = env.get_camera_image()
  obs = env.get_observation()
  prev_obs = copy.deepcopy(obs['image'])

  # Tokenize text and get CLIP features.
  device = next(clip_model.parameters()).device
  text_tokens = clip.tokenize(text).to(device)
  with torch.no_grad():
    text_feats = clip_model.encode_text(text_tokens).float()
  text_feats /= text_feats.norm(dim=-1, keepdim=True)
  text_feats = np.float32(text_feats.cpu())

  # Normalize image and add batch dimension.
  img = obs['image'][None, ...] / 255
  img = np.concatenate((img, coords[None, ...]), axis=3)

  # Run Transporter Nets to get pick and place heatmaps.
  batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
  pick_map, place_map = eval_step(optim.target, batch)
  pick_map, place_map = np.float32(pick_map), np.float32(place_map)
  if unct != None:
    pick_map = (pick_map - pick_map.min())/(pick_map.max()-pick_map.min())
    place_map = (place_map - place_map.min())/(place_map.max()-place_map.min())
    hpick_map = maps['pick']
    hplace_map = maps['place']
    hpick_map, hplace_map = np.float32(hpick_map), np.float32(hplace_map)
    pick_map = (1-unct['obj2'])*pick_map + unct['obj2']*hpick_map
    place_map = (1-unct['sub2'])*place_map + unct['sub2']*hplace_map

  # Get pick position.
  pick_max = np.argmax(np.float32(pick_map)).squeeze()
  pick_yx = (pick_max // 224, pick_max % 224)
  pick_yx = np.clip(pick_yx, 20, 204)
  pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

  # Get place position.
  place_max = np.argmax(np.float32(place_map)).squeeze()
  place_yx = (place_max // 224, place_max % 224)
  place_yx = np.clip(place_yx, 20, 204)
  place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

  # Step environment.
  act = {'pick': pick_xyz, 'place': place_xyz}
  obs, _, _, _ = env.step(act)

  # Show pick and place action.
  # plt.title(text)
  # plt.imshow(prev_obs)
  # plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
  # plt.show()

  # Show debug plots.
  # plt.subplot(1, 2, 1)
  # plt.title('Pick Heatmap')
  # plt.imshow(pick_map.reshape(224, 224))
  # plt.subplot(1, 2, 2)
  # plt.title('Place Heatmap')
  # plt.imshow(place_map.reshape(224, 224))
  # plt.show()

  # Show video of environment rollout.
  # debug_clip = ImageSequenceClip(env.cache_video, fps=25)
  # display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
  # env.cache_video = []

  # Show camera image after pick and place.
  # plt.subplot(1, 2, 1)
  # plt.title('Before')
  # plt.imshow(before)
  # plt.subplot(1, 2, 2)
  # plt.title('After')
  # after = env.get_camera_image()
  # plt.imshow(after)
  # plt.show()

  # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
  return obs
