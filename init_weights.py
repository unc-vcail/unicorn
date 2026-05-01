"""
Initialises a UNICORN ControlNet checkpoint from a pretrained SD1.5 checkpoint.

Usage:
    python init_weights.py <input_sd15.ckpt> <output_unicorn_init.ckpt>
"""

import sys
import os

assert len(sys.argv) == 3, 'Usage: init_weights.py <input_ckpt> <output_ckpt>'

input_path  = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path directory does not exist.'

import torch
from share import *
from cldm.model_loader import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name) or name[:len(parent_name)] != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/config.yaml')

pretrained_weights = torch.load(input_path, map_location='cuda', weights_only=False)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()
target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_model.control_')
    copy_k = ('model.diffusion_' + name) if is_control else k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'New weight (randomly initialised): {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
