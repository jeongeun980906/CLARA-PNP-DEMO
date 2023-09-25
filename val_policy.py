import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
clip_device = "cuda:2"

import json
from clipport.model import TransporterNets, n_params
from clipport.train import train_step, eval_step
from clipport.run import run_cliport

import clip
from vild.forward import vild

from eval.run import run_with_policy

from env.env import PickPlaceEnv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip
from flax.training import checkpoints
import flax
import jax
import os, copy
import jax.numpy as jnp
from env.success import *
import os
import random
import argparse

def main(args):
    keys = ['seen', 'unseen', 'ambiguous', 'ood']

    with open('./data/pick_and_place.json','r') as f:
        data = json.load(f)

    unct_type = args.method
    if args.ab:
        with open('./res/pick_and_place_unct_chat_wp.json'.format(args.ab),'r') as f:
            res = json.load(f)
    elif not args.iter:
        if args.lm == 'gpt':
            with open('./res/pick_and_place_unct_{}.json'.format(unct_type),'r') as f:
                res = json.load(f)

        elif args.lm == 'chat':
            with open('./res/pick_and_place_unct_chat.json','r') as f:
                res = json.load(f)
        elif args.lm == 'llama':
            with open('./res/pick_and_place_unct_llama_{}.json'.format(unct_type),'r') as f:
                res = json.load(f)
        else:
            raise NotImplementedError

    model_name = args.lm
    if args.base:
        with open('./res/pick_and_place_unct_{}_base.json'.format(model_name),'r') as f:
            res = json.load(f)
    
    if args.method == 1 and not model_name == 'chat':
        unct_type = 'ent'
    elif args.method == 3 and not model_name == 'chat':
        unct_type = 'norm_ent'
    elif args.method == 4 and not model_name == 'chat':
        unct_type = 'se_ent'
    elif args.method == 2 or model_name == 'chat':
        unct_type = 'var'
    elif args.method == 7:
        unct_type = 'lu'
    else: 
        raise NotImplementedError
    if args.fewshot:
        unct_type = 'fewshot'

    if args.iter:
        with open('./res/pick_and_place_unct_{}_{}_inter.json'.format(model_name, unct_type),'r') as f:
            res = json.load(f)

    # Initialize model weights using dummy tensors.
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    gpus = jax.devices('gpu')
    print(gpus)
    # init_params = jax.jit(init_params,device=gpus[1])
    print(f'Model parameters: {n_params(init_params):,}')
    optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)


    ckpt_path = f'ckpt_{40000}'
    optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    print('Loaded:', ckpt_path)


    clip_model, clip_preprocess = clip.load("ViT-B/32",device = 'cuda')
    clip_model.eval()
    random.seed(0)
    np.random.seed(0)

    env = PickPlaceEnv()
    coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
    coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)

    try:
        os.mkdir("./res/{}_{}".format(model_name, unct_type))
    except:pass
    try:
        os.mkdir("./res/{}_base".format(model_name))
    except:pass
    try:
        os.mkdir("./res/{}_{}_inter".format(model_name, unct_type))
    except:pass

    for new_name, val in res.items():
        name = new_name.split("_")
        key = name[0]
        name = name[0]+"_" +name[1]
        task = val['name']
        word_success = int(val['success'])
        objs = val['gt_objects']
        found_objects = val['found_objects']
        gt_objects = val['gt_objects']
        places = []
        picks = []
        for obj in objs:
            if 'bowl' in obj:
                places.append(obj)
            else:
                picks.append(obj)
        config = {'pick':picks, 'place':places}
        obs = env.reset(config)
        if args.iter:
            steps_text = val['exce']
            try: 
                indx = val['answer']
            except:
                print("No QNA")
                continue
        else:
            steps_text = val['steps']
        if word_success:
            print("!")
            success = run_with_policy(env,steps_text, clip_model, coords, 
                        optim,obs,gt_objects, task = key, gt=True)
            plt.figure()
            plt.title("{}: {}".format(task, success))
            plt.imshow(env.get_camera_image())
            if args.base:
                plt.savefig("./res/{}_base/{}.png".format(model_name,new_name))
            elif args.iter:
                plt.savefig("./res/{}_{}_inter/{}_{}.png".format(model_name, unct_type,new_name,indx))
            else:
                plt.savefig("./res/{}_{}/{}.png".format(model_name, unct_type,new_name))

        else:
            success = 0
        print(success, task, steps_text)
        res[new_name]['poicy_success'] = [success]

    if args.ab:
        with open('./res/pick_and_place_unct_chat_wp.json'.format(args.ab),'w') as f:
            json.dump(res,f,indent=4)
    elif args.base:
        with open('./res/pick_and_place_unct_{}_base.json'.format(model_name),'w') as f:
            json.dump(res,f,indent=4)
    elif args.inter:
        with open('./res/pick_and_place_unct_{}_{}_inter.json'.format(model_name, unct_type),'w') as f:
            json.dump(res,f,indent=4)
    else:
        with open('./res/pick_and_place_unct_{}_{}.json'.format(model_name, unct_type),'w') as f:
            json.dump(res,f,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--lm', type=str, default='gpt')
    parser.add_argument("--inter", default=False, action='store_true')
    parser.add_argument("--base", default=False, action='store_true')
    parser.add_argument("--fewshot", default=False, action='store_true')
    parser.add_argument("--ab", type=int, default=0)
    args = parser.parse_args()
    main(args)