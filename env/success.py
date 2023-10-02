import math
import imageio
import numpy as np
'''
This file includes code derived from [saycan] by Google LLC.
Copyright 2022 Google LLC.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Source Code: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
'''

def save_img(env, image_path):
    img_top = env.get_camera_image_top()
    imageio.imsave(image_path, img_top)

def success_detector(found_objects,boxes,steps_text):
    pick, place = steps_text.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
    pick_idx = found_objects.index(pick)
    place_idx = found_objects.index(place)
    place_coord = boxes[place_idx]
    pick_coord = boxes[pick_idx]
    pick_x, pick_y = (pick_coord[0] + pick_coord[2]) / 2, (pick_coord[1] + pick_coord[3]) / 2
    place_x, place_y = (place_coord[0] + place_coord[2]) / 2, (place_coord[1] + place_coord[3]) / 2
    if math.sqrt((pick_x-place_x)**2+(pick_y - place_y)**2) < 10:
        return True
    else: return False

def success_detector_gt(env,step_text,task=None):
    print(task)
    pick, place = step_text.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
    if task == 'ood':
        return True
    pick_coord = env.obj_xyz(pick)
    place_coord = env.obj_xyz(place)
    print(pick_coord, place_coord)
    if np.linalg.norm(pick_coord[:-1] - place_coord[:-1]) < 0.2:
        return True
    else: return False