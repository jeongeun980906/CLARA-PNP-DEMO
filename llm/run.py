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