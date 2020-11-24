import gym
import numpy as np
import random
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from NormalizedEnvironment import NormalizedEnv
from Model import Actor
import time
from torch.distributions.uniform import Uniform

from InvPendulum import InvPendulumEnv

if __name__ == '__main__':
	num_warmup = 1000
	num_train = 200000
	num_eval = 0
	buffer_length = 600000

	# env = NormalizedEnv(gym.make('Pendulum-v0'))
	GODOT_BIN_PATH = "InvPendulum/InvPendulum.x86_64"
	env_abs_path = "InvPendulum/InvPendulum.pck"
	env = NormalizedEnv(InvPendulumEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path, render=True))

	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]
	policy = Actor(num_states, num_actions)
	policy.load_state_dict(torch.load('./policy.pkl'))

	state = env.reset()
	state = state.to(dtype=torch.float32)

	traced_policy = torch.jit.trace(policy, state)
	print(traced_policy.graph)
	print(traced_policy.code)
	traced_policy.save('ddpg_policy.jit')

	for step in range(1000):
	
		action = policy(state)
		#			torch.tensor([1.0 for i in range(num_actions)])).sample().to(device='cuda')
		time.sleep(0.02)
		# state_next, reward, term, _ = env.step(action.cpu().numpy())
		state_next, reward, term, _ = env.step(action)
		# state_next = torch.from_numpy(state_next).to(dtype=state.dtype, device=state.device)
		state = state_next.to(dtype=state.dtype, device=state.device)
		
	env.close()