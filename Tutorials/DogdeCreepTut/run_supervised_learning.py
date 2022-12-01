from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from collections import deque, defaultdict
from absl import flags

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import retro
import gym
import os
import argparse
import cv2

import network

parser = argparse.ArgumentParser(description='Sonic Supervised Learning')
parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
parser.add_argument('--replay_path', type=str, help='root directory of dataset')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

arguments = parser.parse_args()

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


workspace_path = arguments.workspace_path
replay_path = arguments.replay_path

writer = tf.summary.create_file_writer(workspace_path + "/tensorboard")

action_conversion_table = {
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']        
                '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['DOWN']
                '[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['A', 'DOWN']

                '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['A']
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['A', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['']
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']

                '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT']
                '[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['LEFT'] 

                '[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['A', 'RIGHT', 'DOWN']
                '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['RIGHT']
                '[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['A', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']

                '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT', 'DOWN']
                '[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['B', 'LEFT', 'RIGHT']
                '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # = ['UP']

                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT', 'DOWN']
                '[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # = ['DOWN', 'LEFT', 'RIGHT']
                '[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['A', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']

                '[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT’]

                '[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']

                '[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT', 'DOWN']
                '[1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['B', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT']
                '[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']

                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']

                '[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['DOWN', RIGHT']
              }

# ['']:                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['LEFT']:                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['RIGHT']:                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['B']:                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['DOWN']:                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['A']:                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['UP']:                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ['B', 'LEFT']:             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['B', 'RIGHT']:            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['B', 'DOWN']:             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['A', 'LEFT']:             [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['A', 'RIGHT']:            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['A', 'DOWN']:             [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['DOWN', 'RIGHT']:         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['DOWN', 'LEFT']:          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['LEFT', 'RIGHT']:         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['B', 'RIGHT', 'DOWN']:    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['B', 'LEFT', 'DOWN'] :    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['A', 'LEFT', 'DOWN'] :    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['A', 'LEFT', 'RIGHT']:    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['A', 'RIGHT', 'DOWN']:    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['B', 'LEFT', 'RIGHT']:    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['DOWN', 'LEFT', 'RIGHT']: [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


possible_action_list = [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
                       ]

stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']
stage_len = len(stage_name_list)

class TrajetoryDataset(tf.data.Dataset):
  def _generator(num_trajectorys):
    env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)

    while True:
        replay_file_path_list = glob.glob(replay_path + '/*.bk2')
        replay_name = random.choice(replay_file_path_list)
        replay_name = replay_name.split('/')[-1]

        #stage_name = replay_name.split('-')
        #stage_name = stage_name[2].split('.')[0]
        #stage_index = stage_name_list.index(stage_name)
        #print("stage_index: ", stage_index)

        replay = retro.Movie(os.path.join(replay_path, replay_name))
        replay.step()

        env.initial_state = replay.get_state()
        obs = env.reset()

        #obs = 0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]
        #obs[obs < 100] = 0
        #obs[obs >= 100] = 255
        obs = cv2.resize(obs, dsize=(84,84), interpolation=cv2.INTER_AREA) / 255.0

        #obs_t = np.stack((obs, obs, obs, obs), axis=2)

        action_index = 0

        obs_list, action_list = [], []

        step_num = 0

        print('stepping replay')
        while replay.step():
            keys = []
            for i in range(len(env.buttons)):
                key = int(replay.get_key(i, 0))
                keys.append(key)

            converted_action = action_conversion_table[str(keys)]
            #print("converted_action: ", converted_action)

            action_index = possible_action_list.index(converted_action)
            #print("action_index: ", action_index)

            #obs_resized = cv2.resize(obs, dsize=(64,64), interpolation=cv2.INTER_AREA)
            #obs_resized = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2RGB)
            #obs_resized = np.reshape(obs_resized,(64,64,3)) / 255.0
            #stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
            #stage_layer[:, :, stage_index] = 1.0

            #obs_concated = np.concatenate((obs_resized, stage_layer), axis=2)
            #obs_concated = obs_t

            obs_list.append(obs)
            action_list.append(np.array([action_index]))

            next_obs, rew, done, info = env.step(converted_action)

            #next_obs = 0.299*next_obs[:,:,0] + 0.587*next_obs[:,:,1] + 0.114*next_obs[:,:,2]
            #next_obs[next_obs < 100] = 0
            #next_obs[next_obs >= 100] = 255
            next_obs = cv2.resize(next_obs, dsize=(84,84), interpolation=cv2.INTER_AREA) / 255.0


            obs = next_obs
            #x_t1 = np.reshape(next_obs, (84,84,1))
            #obs_t = np.append(x_t1, obs_t[:, :, :3], axis=2)
            #env.render()

            saved_state = env.em.get_state()
            step_num += 1

            #if step_num == 100:
            #    break

        yield (obs_list, action_list)
        '''
        list_len = len(obs_list)

        sample = random.random()
        print("sample: ", sample)
        if sample > 0.5:
            yield (obs_list[0:int(list_len / 2)], action_list[0:int(list_len / 2)])
        else:
            yield (obs_list[int(list_len / 2):], action_list[int(list_len / 2):])
        '''
        break

  def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.float32, tf.dtypes.int32),
          args=(num_trajectorys,)
      )

dataset = tf.data.Dataset.range(1).interleave(TrajetoryDataset, 
  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

num_actions = len(possible_action_list)
num_hidden_units = 1024

#model = tf.keras.models.load_model('MineRL_SL_Model')
model = network.ActorCritic(num_actions, num_hidden_units)

if arguments.pretrained_model != None:
    print("Load Pretrained Model")
    model.load_weights("model/" + arguments.pretrained_model)

    
cce_loss = tf.keras.losses.CategoricalCrossentropy()
cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001)


@tf.function
def supervised_replay(replay_obs_list, replay_act_list, memory_state, carry_state):
    replay_obs_array = tf.concat(replay_obs_list, 0)
    replay_act_array = tf.concat(replay_act_list, 0)

    replay_memory_state_array = tf.concat(memory_state, 0)
    replay_carry_state_array = tf.concat(carry_state, 0)

    memory_state = replay_memory_state_array
    carry_state = replay_carry_state_array

    batch_size = replay_obs_array.shape[0]
    #tf.print("batch_size: ", batch_size)
    
    with tf.GradientTape() as tape:
        act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            prediction = model(tf.expand_dims(replay_obs_array[i,:,:,:], 0), memory_state, carry_state, 
                                training=True)
            act_pi = prediction[0]
            memory_state = prediction[2]
            carry_state = prediction[3]
        
            act_probs = act_probs.write(i, act_pi[0])

        act_probs = act_probs.stack()

        #tf.print("replay_act_array: ", replay_act_array)
        #tf.print("tf.argmax(act_probs, 1): ", tf.argmax(act_probs, 1))

        replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
        replay_act_array_onehot = tf.reshape(replay_act_array_onehot, (batch_size, num_actions))
        act_loss = cce_loss_logits(replay_act_array_onehot, act_probs)

        #tf.print("act_loss: ", act_loss)
        regularization_loss = tf.reduce_sum(model.losses)
        total_loss = act_loss + 1e-5 * regularization_loss
    
        #tf.print("total_loss: ", total_loss)
        #tf.print("")
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, memory_state, carry_state


def supervised_train(dataset, training_episode):
    for batch in dataset:
        episode_size = batch[0].shape[1]
        print("episode_size: ", episode_size)
    
        replay_obs_list = batch[0][0]
        replay_act_list = batch[1][0]
     
        memory_state = np.zeros([1,128], dtype=np.float32)
        carry_state =  np.zeros([1,128], dtype=np.float32)
        step_length = 32
        total_loss = 0
        for episode_index in range(0, episode_size, step_length):
            obs = replay_obs_list[episode_index:episode_index+step_length,:,:,:]
            act = replay_act_list[episode_index:episode_index+step_length,:]
            
            #print("episode_index: ", episode_index)
            if len(obs) != step_length:
                break
            
            total_loss, next_memory_state, next_carry_state = supervised_replay(obs, act,
                                                                                memory_state, carry_state)
            memory_state = next_memory_state
            carry_state = next_carry_state
        
            #print("total_loss: ", total_loss)
            #print("")
            
        with writer.as_default():
            #print("training_episode: ", training_episode)
            tf.summary.scalar("total_loss", total_loss, step=training_episode)
            writer.flush()

        if training_episode % 100 == 0:
            model.save_weights(workspace_path + '/model/supervised_model_' + str(training_episode))
            
        
for training_episode in range(0, 2000000):
    #print("training_episode: ", training_episode)
    supervised_train(dataset, training_episode)