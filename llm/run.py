'''
This file includes code derived from [saycan] by Google LLC.
Copyright 2022 Google LLC.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Source Code: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb

Modifications made on 2023-10-03:
Extracted the run part of the original code.
'''
import numpy as np
from llm.affordance import affordance_score2
import matplotlib.pyplot as plt
from clipport.run import run_cliport
from llm.helper import *
import copy
from env.success import *

def run(lm_planner,found_objects,max_tasks=5):
    num_tasks = 0
    steps_text = []
    uncts = []
    done = False
    while not done:
        num_tasks += 1
        if num_tasks > max_tasks:
            break
        tasks, scores , unct = lm_planner.plan_with_unct()
        for t in tasks:
            if 'done' in t:
                done = True
                break
        if done:
            break
        # print(tasks)
        if tasks != None:
            if len(scores)>0:
                scores = np.asarray(scores)
                idxs= np.argsort(scores)
                for idx in idxs[::-1]:
                    try:
                        aff = affordance_score2(tasks[idx], found_objects)
                    except:
                        print(tasks[idx])
                        aff = 0
                    if aff > 0:
                        break
                if aff == 2: 
                    done=True 
                    lm_planner.append(None, None, tasks[idx])
                    # steps_text.append("done()")
                    # uncts.append(unct)
                    break
                selected_task = tasks[idx]
                print(selected_task, aff, unct)
                steps_text.append(selected_task)
                
                uncts.append(unct)
                lm_planner.append(None, None, selected_task)
            
        else:
            break
    return steps_text, uncts


def run_with_policy(env,steps_text,clip_model,coords,optim,obs, found_objects,
                vild_model = None, image_path = None, task = None, gt = False):
    # print('Initial state:')
    # plt.imshow(env.get_camera_image())
    # found_objects, boxes = vild_model.infer(image_path)
    # category_name_string = ";".join(found_objects)
    # vild_model.category_name_string = copy.deepcopy(category_name_string)
    total_success = True
    # _,_ = vild_model.infer(image_path)
    for i, step in enumerate(steps_text):
        if step == '' or step == 'done()':
            break
        nlp_step = step_to_nlp(step)
        print('GPT-3 says next step:', nlp_step)
        success = False
        count = 0
        aff = affordance_score2(step,found_objects)
        if aff != 1 and task != 'ood':
            total_success *= success
            break
        while not success:
            obs = run_cliport(env,clip_model,coords, optim, obs, nlp_step)
            if gt:
                # try:
                success = success_detector_gt(env,step,task)
                # except:
                #     success = False
                count += 10
            else:
                save_img(env, image_path)
                found_objects, boxes = vild_model.infer(image_path)
                success = success_detector(found_objects,boxes,step)
                count += 1
            if count > 3:
                break
        total_success *= success
        if total_success == 0:
            break
    # Show camera image after task.
    # print('Final state:')
    # plt.imshow(env.get_camera_image())
    # final_image = env.get_camera_image()

    return total_success