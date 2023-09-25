import openai
import random
import spacy
import scipy
import copy
import numpy as np
import json
from llm.affordance import affordance_scoring, affordance_score2
from llm.lnct_score import lm_planner_unct # GPT3
# from llm.llama import llama_unct
from llm.chat import lm_planner_unct_chat
from eval.run import run
import argparse
import numpy as np
import random
import cv2
from env.success import *
max_tasks = 5

def set_openai_api_key_from_txt(key_path='./key.txt',VERBOSE=True):
    """
        Set OpenAI API Key from a txt file
    """
    with open(key_path, 'r') as f: 
        OPENAI_API_KEY = f.read()
    openai.api_key = OPENAI_API_KEY
    if VERBOSE:
        print ("OpenAI API Key Ready from [%s]."%(key_path))
    
def run_baseline(args):
    random.seed(0)
    np.random.seed(0)
    res = {}
    unct_type = args.method
    lm_type = args.lm
    if lm_type == 'gpt3':
        lm_planner = lm_planner_unct(type=unct_type)
    elif lm_type == 'chat':
        lm_planner = lm_planner_unct_chat()

    with open('./res/pick_and_place_unct_vild.json','r') as f:
        data = json.load(f)
    for new_name, task in data.items():
        name = new_name.split("_")
        idx = name[-1]
        task_cat = name[0]
        name = name[0]+"_"+ name[1]
        goal = task['name']
        found_objects = task['found_objects']
        # print(goal)
        flag = False
        for i in range(int(idx)):
            temp_name = new_name[:-1] + str(i)
            temp_objects = data[temp_name]['found_objects']
            if len(temp_objects) == len(found_objects):
                flag = True
                for t in temp_objects:
                    if t not in found_objects:
                        flag = False
                        break
                if flag:
                    print("overlapped")
                    res[new_name] = copy.deepcopy(res[temp_name])
                    break
            else:
                flag = False
        if not flag:
            lm_planner.reset()
            lm_planner.objects = copy.deepcopy(found_objects)
            lm_planner.set_goal(goal)
            lm_planner.set_prompt()
            new_step = lm_planner.infer_wo_unct(found_objects,task_cat,stop=False)
            print(new_step)
            res[new_name]= {
                    'name':goal,
                    'gt_objects': task['gt_objects'],
                    'found_objects': found_objects,
                    'steps':new_step
                }
    with open('./res/pick_and_place_unct_{}_base.json'.format(lm_type),'w') as f:
        json.dump(res, f, indent=4)
    return

def main(args):
    set_openai_api_key_from_txt()
    random.seed(0)
    np.random.seed(0)
    with open('./res/pick_and_place_unct_vild.json','r') as f:
        data = json.load(f)

    unct_type = args.method
    lm_type = args.lm
    if lm_type == 'gpt3':
        lm_planner = lm_planner_unct(type=unct_type)
    elif lm_type == 'chat':
        lm_planner = lm_planner_unct_chat()
        print('chat')
    # elif lm_type == 'llama':
    #     lm_planner = llama_unct()
    max_tasks = 5
    if args.resume:
        print("resume")
        with open('./res/pick_and_place_unct_{}.json'.format(unct_type),'r') as f:
            res = json.load(f)
            key = 'ood'
    else:
        res = {}
    text_only = args.text
    for new_name, task in data.items():
        name = new_name.split("_")
        idx = name[-1]
        name = name[0]+"_"+ name[1]
        if args.resume and key not in name:
            continue
        goal = task['name']
        if text_only:
            found_objects = task['gt_objects']
            if int(idx) == 0:
                flag = False
            else:
                flag = True
        else:
            found_objects = task['found_objects']
            # print(goal)
            flag = False
            for i in range(int(idx)):
                temp_name = new_name[:-1] + str(i)
                temp_objects = data[temp_name]['found_objects']
                if len(temp_objects) == len(found_objects):
                    flag = True
                    for t in temp_objects:
                        if t not in found_objects:
                            flag = False
                            break
                    if flag:
                        print("overlapped")
                        res[new_name] = copy.deepcopy(res[temp_name])
                        break
                else:
                    flag = False
        if not flag:
            print(goal)
            lm_planner.objects = copy.deepcopy(found_objects)
            lm_planner.set_goal(goal)
            steps_text, uncts = run(lm_planner, found_objects,max_tasks)
            res[new_name]= {
                'name':goal,
                'gt_objects': task['gt_objects'],
                'found_objects': found_objects,
                'steps':steps_text,
                'uncertainties':uncts
            }
        lm_planner.reset()

    if text_only:
        if lm_type == 'chat':
            with open('./res/pick_and_place_unct_chat_text.json'.format(unct_type),'w') as f:
                json.dump(res, f, indent=4)
        else:
            with open('./res/pick_and_place_unct_{}_text.json'.format(unct_type),'w') as f:
                json.dump(res, f, indent=4)
    else:
        if lm_type == 'chat':
            with open('./res/pick_and_place_unct_chat_7.json'.format(unct_type),'w') as f:
                json.dump(res, f, indent=4)
        else:
            with open('./res/pick_and_place_unct_{}.json'.format(unct_type),'w') as f:
                json.dump(res, f, indent=4)
    print("saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",default=1, type = int)          # extra value
    parser.add_argument("--lm", default='gpt3', type=str)           # existence/nonexistence
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--text", default=False, action='store_true')
    parser.add_argument("--baseline", default=False, action='store_true')
    args = parser.parse_args()
    if args.baseline:
        run_baseline(args)
    else:
        main(args)