import fp_env
from collections import OrderedDict
import torch
import circuit_dataloader
import utils
import os
from tqdm import tqdm
from arguments import get_args
import math
import tianshou
from tianshou.env import DummyVectorEnv
from tianshou.data import VectorReplayBuffer, Batch
from einops import rearrange
from copy import deepcopy

# from tianshou.policy import PPOPolicy
from policy import PPOPolicy, ClipLossCoef, EntropyLossCoef, load_ppo_policy

# from tianshou.trainer import OnpolicyTrainer
from trainer import OnpolicyTrainer

# from tianshou.data import Collector
from collector import Collector, get_statistics

import model
import numpy as np
from utils import TensorboardWriter, save_final_floorplan
import pandas as pd
from collections import defaultdict
import subprocess
import time
import pdb


args = get_args()

utils.setup_seed(args.seed)
num_grid_x = args.num_grid_x
num_grid_y = args.num_grid_y
result_dir = args.result_dir
fig_dir = os.path.join(result_dir, "fig")
save_checkpoint_dir = os.path.join(result_dir, "checkpoint")

num_env = args.num_env
num_env_test = args.num_env_test

wiremask_bbo = args.wiremask_bbo
device = torch.device(args.device)
episode_per_collect_per_env = args.episode_per_collect_per_env
save_batch_dir = os.path.join(result_dir, "batch") if args.save_batch else None

args.circuit = "nangate45_ariane136"
args.impl = "nangate45"
args.design = "ariane136"


# fp_info
fp_info, df_partner = circuit_dataloader.construct_fp_info_func(args.circuit, args.area_util, num_grid_x, num_grid_y, 
                                                    args.num_alignment, args.alignment_rate, args.alignment_sort, args.num_preplaced_module, args.add_virtual_block, args.num_layer, True, False, False, 0, 0, [0.0,0.0])

#pdb.set_trace()

episode_len = fp_info.movable_block_num
grid_hpwl, weigted_grid_hpwl, original_hpwl = fp_info.calc_hpwl()
via = fp_info.calc_via()
area_ratio = fp_info.calc_area_ratio()
num_ratio = fp_info.calc_num_ratio()

# output the floorplan
save_final_floorplan("openroad.png", fp_info, args.impl, args.design, 0, 0)

print("original hpwl: ", original_hpwl)
print("via: ", via)
print("area_ratio: ", area_ratio)
print("num_ratio: ", num_ratio)
