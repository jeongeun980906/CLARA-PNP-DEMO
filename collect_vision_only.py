'''
This file includes code derived from [saycan] by Google LLC.
Copyright 2022 Google LLC.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Source Code: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb

Modifications made on 2023-10-03:
Extracted the vision part of the original code.
'''
import clip
from env.env import PickPlaceEnv
import numpy as np
import random
import clip
import cv2
from env.success import *
from vild.forward import vild
import json, copy, os
import matplotlib.pyplot as plt

category_names = ['blue block',
                  'red block',
                  'green block',
                  'orange block',
                  'yellow block',
                  'purple block',
                  'pink block',
                  'cyan block',
                  'brown block',
                  'gray block',

                  'blue bowl',
                  'red bowl',
                  'green bowl',
                  'orange bowl',
                  'yellow bowl',
                  'purple bowl',
                  'pink bowl',
                  'cyan bowl',
                  'brown bowl',
                  'gray bowl']

other_objects = [
    'apple', 'banana', 'camera', 'bottle', 'can'
]
other_objects += copy.deepcopy(category_names)

category_name_string = ";".join(category_names)
other_category_name_string = ";".join(other_objects)
max_boxes_to_draw = 8 #@param {type:"integer"}

# Extra prompt engineering: swap A with B for every (A, B) in list.
prompt_swaps = [('block', 'cube')]

nms_threshold = 0.2 #@param {type:"slider", min:0, max:0.9, step:0.05}
min_rpn_score_thresh = 0.2  #@param {type:"slider", min:0, max:1, step:0.01}
min_box_area = 100 #@param {type:"slider", min:0, max:10000, step:1.0}
max_box_area = 5000  #@param {type:"slider", min:0, max:10000, step:1.0}
vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
image_path = "./2db.png"

random.seed(0)
np.random.seed(0)

clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()
vild_model = vild(clip_model,category_name_string, vild_params)

env = PickPlaceEnv()
with open('./data/pick_and_place.json','r') as f:
    data = json.load(f)
res = {}
keys = ['seen', 'unseen', 'ambiguous']

try:
    os.mkdir("./res/init")
except:
    pass

for j in range(3):
    for k in keys:
        vild_model.category_name_string = copy.deepcopy(category_name_string)
        for i in range(20):
            if k == 'ambiguous' and i>14: continue
            name = "{}_{}".format(k, i)
            task = data[name]
            goal = task['name']
            GT = task['gt']
            new_name = "{}_{}".format(name, j)
            gt_objects = task['objects']
            places = []
            picks = []
            for obj in gt_objects:
                if 'bowl' in obj:
                    places.append(obj)
                else:
                    picks.append(obj)
            config = {'pick':picks, 'place':places}
            obs = env.reset(config)
            save_img(env, image_path)
            found_objects,_ = vild_model.infer(image_path, plot_on = True, 
                            name = "{}_{}".format(name,j))
            res[new_name]= {
                    'name':goal,
                    'gt_objects': gt_objects,
                    'found_objects': found_objects,
                    'gt': GT
            }

with open('./res/pick_and_place_unct_vild.json','w') as f:
    json.dump(res, f, indent=4)
