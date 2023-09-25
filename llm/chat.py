import openai
import spacy
import scipy
import random
import numpy as np
import copy
from llm.affordance import affordance_scoring, affordance_score2
import time

class lm_planner_unct_chat():
    def __init__(self, example = False):
        self.few_shots = [
        """
        task: move all the blocks to the top left corner.
        scene: objects = [red block, yellow block, blue block, green bowl]
        robot action: robot.pick_and_place(blue block, top left corner)
        robot action: robot.pick_and_place(red block, top left corner)
        robot action: robot.pick_and_place(yellow block, top left corner)
        robot action: done()
        """
        ,
        """
        task: put the yellow one the green thing.
        scene: objects = [red block, yellow block, blue block, green bowl]
        robot action: robot.pick_and_place(yellow block, green bowl)
        robot action: done()
        """
        ,
        """
        task: move the light colored block to the middle.
        scene: objects = [yellow block, blue block, red block]
        robot action: robot.pick_and_place(yellow block, middle)
        robot action: done()
        """
        ,
        """
        task: stack all the blocks.
        scene: objects = [blue block, green bowl, red block, yellow bowl, green block]
        robot action: robot.pick_and_place(green block, blue block)
        robot action: robot.pick_and_place(red block, green block)
        done()
        """
        ,
        """
        task: group the blue objects together.
        scene: objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
        robot action: robot.pick_and_place(blue block, blue bowl)
        robot action: done()
        """
        ,
        """
        task: put all blocks in the green bowl.
        scene: objects = [red block, blue block, green bowl, blue bowl, yellow block]
        robot action: robot.pick_and_place(red block, green bowl)
        robot action: robot.pick_and_place(blue block, green bowl)
        robot action: robot.pick_and_place(yellow block, green bowl)
        robot action: done()
        """
        # ,
        # """
        # task: sort all the blocks into their matching color bowls.
        # scene: objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
        # robot action: robot.pick_and_place(green block, green bowl)
        # robot action: robot.pick_and_place(red block, red bowl)
        # robot action: robot.pick_and_place(yellow block, yellow bowl)
        # robot action: done()
        # """
        # ,
        # """
        # task: put all the blocks in different corners.
        # scene: objects = [yellow block, green block, red bowl, red block, blue block]
        # robot action: robot.pick_and_place(blue block, top right corner)
        # robot action: robot.pick_and_place(green block, bottom left corner)
        # robot action: robot.pick_and_place(red block, top left corner)
        # robot action: robot.pick_and_place(yellow block, bottom right corner)
        # robot action: done()
        # """
        ]
        if example:
            self.few_shots[3] = """
        task: stack all the blocks.
        scene: objects = [blue block, green bowl, red block, yellow bowl, green block]
        robot thought: This is code is uncertain because I don't know which block to pick up first.
        robot thought: What can I ask to the user?
        question: Which block should I pick up first?
        answer: green block
        robot action: robot.pick_and_place(green block, blue block)
        robot action: robot.pick_and_place(red block, green block)
        done()
            """
        self.new_lines = ""
        self.nlp = spacy.load('en_core_web_lg')
        self.type = 7
        self.verbose = True
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']

        self.set_func()
        
    def set_func(self):
        if self.type == 2 or self.type == 4:
            self.plan_with_unct = self.plan_with_unct_type2
        elif self.type == 7:
            self.plan_with_unct = self.plan_with_unct_type6
        else:
            raise NotImplementedError

    def plan_with_unct_type6(self, verbose = False):
        self.set_prompt()
        object = ""
        subject = ""
        # Only one beam search? -> N samples
        while (len(object) < 3 or len(subject)< 3):
            object, subject = self.inference()
            # print(object_probs,subject_probs)
        temp = 'robot action: robot.pick_and_place({}, {})'.format(object[0],subject[0])
        temp += "robot thought: Is this certain enough please answer in yes or no?\nrobot thought: "
        inp = copy.deepcopy(self.prompt)
        inp += temp
        ans = ""
        time.sleep(5)
        while len(ans)<3:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": inp}], 
                    temperature = 0.8, top_p = 1, n = 3, stop='\n'
                    )
                ans = response['choices'][0]['message']['content']
            except:
                time.sleep(300)
        
        if ans[0] ==" ":
            ans = ans[1:]
        print(ans)
        if 'yes' in ans.lower().replace(".","").replace(",",""):
            unct = 0 
        else:
            unct =  1
        return ['robot action: robot.pick_and_place({}, {})'.format(object[0],subject[0])], [1], {'total':unct}


    def plan_with_unct_type2(self, verbose= False):
        obj_cand = []
        subj_cand = []
        self.verbose = verbose
        goal_num = 5
        if self.type == 4:
            self.set_prompt()
        while len(obj_cand) <1 or len(subj_cand)<1:
            for _ in range(goal_num):
                if self.type != 4:
                    self.sample_prompt()
                object, subject = self.inference()
                if len(object) != 0:
                    obj_cand += object
                if len(subject) != 0:
                    subj_cand += subject
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.pick_and_place({}, {})'.format(x,y)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1
        scores = [s/sum(scores) for s in scores]
        # print(obj_cand,subj_cand)
        obj2 = self.get_word_diversity(obj_cand)
        sub2 = self.get_word_diversity(subj_cand)
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /10,
            'sub': sub2/10,
            'total': (obj2+sub2)/10
        }

        return tasks, scores, unct

    
    def set_goal(self, goal):
        self.goal = goal

    def set_prompt(self,choices=None):
        des = ""
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        temp = ""
        for e, obj in enumerate(self.objects):
            temp += obj
            if e != len(self.objects)-1:
                temp += ", "
        
        des += "task: considering the ambiguity of the goal,"
        des += self.goal
        # des += "\n where the place object is not dependent from the selected pick object \n"
        des += "scene: objects = [" + temp + "] \n"
        # des += "\n The order can be changed"
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def sample_prompt(self):
        lengs = len(self.few_shots)
        # print(lengs)
        k = random.randrange(4,lengs+1)
        A = np.arange(lengs)
        A = np.random.permutation(A)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        random.shuffle(self.objects)
        self.set_prompt(choices)
        # print(self.prompt)

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\nrobot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += 'robot thought: what can I ask to the user? \nquestion: please' + question


    def inference(self):
        time.sleep(5)
        while True:
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": self.prompt}], 
                temperature = 0.8, top_p = 1, n = 3, stop=')'
                )
                break
            except:
                print("time out?")
                time.sleep(300)
                continue
        # print(response)
        
        objects = []
        subjects = []
        results = response['choices']
        for res in results:
            res = res['message']['content']
            res = res.split("\n")
            if res[0] == "":
                try:
                    res = res[1]
                except:
                    continue
            else:
                res = res[0]
            # print(res)
            if "robot action: done(" in res:
                objects.append("done")
                subjects.append("done")
            if "robot action: robot.pick_and_place" not in res:
                continue
            try:
                pick, place = res.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
            except:
                continue
            pick = pick.split("\n")[-1]
            place = place.split("\n")[0]
            if pick[-1] == " ":
                pick = pick[:-1]
            if place[-1]==" ":
                place = place[:-1]
            if pick[0] == " ":
                pick = pick[1:]
            if place[0] == " ":
                place = place[1:]
            objects.append(pick)
            subjects.append(place)
        return objects, subjects
        
    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot action: robot.pick_and_place({}, {})".format(object, subject)
        else:
            next_line = "\n" + task
        self.new_lines += next_line

    def get_word_diversity(self, words):
        vecs = []
        size = len(words)
        for word in words:
            vec = self.nlp(word).vector
            vecs.append(vec)
        vecs = np.vstack(vecs)
        dis = scipy.spatial.distance_matrix(vecs,vecs)
        div = np.sum(dis)/((size)*(size-1))
        # print(div, dis)
        return div

    def question_generation(self):
        form = '\nrobot thought: this code is uncertain because '
        self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": inp}], 
            temperature = 1.0, top_p = 1, n = 1, stop=':'
            )
        reason = response['choices'][0]['message']['content'].split('\n')[0]
        print('reason: ',reason)
        inp += reason
        self.new_lines += reason + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": inp}], 
            temperature = 1.0, top_p = 1, n = 1, stop='\n'
            )
        ques = response['choices'][0]['message']['content']
        ques = ques.split('\n')[0]
        print('question: please',ques)
        self.new_lines += ques
        return reason, ques
    
    def answer(self, user_inp):
        self.new_lines += '\nanswer:' + user_inp
        self.new_lines += '\robot thought: continue the previous task based on the question and answer'

    def reset(self):
        self.new_lines = ""

    def infer_wo_unct(self, found_objects, task=None, stop=True):
        done = False
        max_tasks=5
        cont = 0
        res = []
        while not done:
            self.set_prompt()
            if cont > max_tasks:
                break
            while True:
                try:
                    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": self.prompt}], 
                    temperature = 0.8, top_p = 1, n = 3
                    )
                    break
                except:
                    time.sleep(5)
                    continue
            text = response['choices'][0]['message']['content'].split('\n')
            
            for line in text:
                if 'done' in line:
                    done = True
                    break
                if "robot action: robot.pick_and_place" in line:
                    cont += 1
                    try: aff = affordance_score2(line, found_objects)
                    except: aff = 0
                    if aff ==0 and task != 'ood':
                        break
                    else:
                        res.append(line)
                        self.append(None, None, line)
                elif "robot thought:" in line:
                    res.append(line)
                    self.append(None, None, line)
                elif "question:" in line and stop:
                    res.append(line)
                    done = True
                    break
        return res