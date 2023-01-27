import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import cv2
import tensorflow_probability as tfp
import argparse
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from absl import flags

from dodgeCreep import dodgeCreepEnv

import network

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='DodgeCreep Evaluation')

parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--model_name', type=str, help='name of saved model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

arguments = parser.parse_args()

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

workspace_path = arguments.workspace_path

num_actions = 5
num_hidden_units = 512

model = network.ActorCritic(num_actions, num_hidden_units)
#model.load_weights("model/" + arguments.model_name)

seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

reward_sum = 0

for i_episode in range(0, 10000):
    # Create the environment
    GODOT_BIN_PATH = "dodge_the_creeps/DodgeCreep.x86_64"
    env_abs_path = "dodge_the_creeps/DodgeCreep.pck"
    env = dodgeCreepEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path, turbo_mode=True)

    obs = env.reset()

    obs = np.reshape(obs, (128,128,3)) / 255.0
    obs = np.array(obs).astype(np.float32)
    obs_resized = cv2.resize(obs, dsize=(64,64), interpolation=cv2.INTER_AREA)
    obs_resized = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2RGB)
    state = obs_resized

    memory_state = tf.zeros([1,256], dtype=np.float32)
    carry_state = tf.zeros([1,256], dtype=np.float32)
    step = 0
    while True:
        step += 1
        #env.render()

        cv2.imshow("state: ", state)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

        #print("state.shape: ", state.shape)
        action_probs, _, memory_state, carry_state = model(tf.expand_dims(state, 0), memory_state, carry_state)
        #action_dist = tfd.Categorical(logits=action_probs)
        
        #if np.random.rand(1) > e:
        #    action_index = int(action_dist.sample()[0])
        #else:
        #    action_index = random.randint(0, len(possible_action_list) - 1)
        action = random.randint(0, 4)
        #action = int(action_dist.sample()[0])

        next_obs, reward, done, info = env.step(action)

        next_obs = np.reshape(next_obs, (128,128,3)) / 255.0
        next_obs = np.array(next_obs).astype(np.float32)
        next_obs_resized = cv2.resize(next_obs, dsize=(64,64), interpolation=cv2.INTER_AREA)
        next_obs_resized = cv2.cvtColor(next_obs_resized, cv2.COLOR_BGR2RGB)

        next_state = next_obs_resized

        reward_sum += reward

        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

env.close()