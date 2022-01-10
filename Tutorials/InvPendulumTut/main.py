import gym
import numpy as np
import random
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from DDGP import DDPG
from ReplayBuffer import ReplayBuffer
from Noise import OrnsteinUhlenbeckProcess
from Logger import Logger
from NormalizedEnvironment import NormalizedEnv

from InvPendulum import InvPendulumEnv

if __name__ == '__main__':
	num_warmup = 1000
	num_train = 200000
	num_eval = 0
	buffer_length = 600000

	GODOT_BIN_PATH = "InvPendulum/InvPendulum.x86_64"
	env_abs_path = "InvPendulum/InvPendulum.pck"
	env = NormalizedEnv(InvPendulumEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path))

	ddpg = DDPG(env)
	logger = Logger('Logs/debug')
	logger.clean()
	buffer = ReplayBuffer(buffer_length)
	state = env.reset()
	state = state.to(dtype=torch.float32, device='cuda')

	noise = OrnsteinUhlenbeckProcess()
	noise.reset_states()

	loss_critic = []
	loss_policy = []
	rewards = []
	phase_path = []
	average_q = []

	episode = 0
	for training_step in range(num_train + num_warmup + num_eval):
	
		if training_step<=num_warmup:
			action = ddpg.random_action().to(dtype=state.dtype, device=state.device)
		else:
			action = ddpg.select_action(state, noise)

		state_next, reward, term, _ = env.step(action.cpu())
		state_next = state_next.to(dtype=state.dtype, device=state.device)
		buffer.append(state, action, reward, state_next, term)

		if training_step>num_warmup and training_step<=(num_train+num_warmup):
			critic_loss, policy_loss = ddpg.train(buffer)
			loss_critic.append(critic_loss)
			loss_policy.append(policy_loss)
		
		elif training_step>(num_train+num_warmup):
			env.render()
		
		average_q.append(ddpg.get_value(state, action))
		phase_path.append(state.cpu().numpy())
		state = state_next.clone()
		if not reward is None:
			rewards.append(reward)

		if term:
			state = env.reset()
			state = state.to(dtype=torch.float32, device='cuda')
			noise.reset_states()
			print(f'Step: {training_step} / Episode: {episode} / Score: {np.sum(rewards)}')
			
			ah, rh, trh = buffer.get_histograms()
			if len(loss_critic) > 0:
				lc = np.average(loss_critic)
				lp = np.average(loss_policy)
				critic_act_h = ddpg.critic.get_act_hist()
				policy_act_h = ddpg.policy.get_act_hist()
			else:
				lc = 0.0
				lp = 0.0
				critic_act_h = np.zeros((1)), np.zeros((1))
				policy_act_h = np.zeros((1)), np.zeros((1))

			av_rw = np.average(rewards)
			min_rw = np.min(rewards)
			max_rw = np.max(rewards)
			entropy = ddpg.get_entropy(buffer)
			avq = np.average(average_q)

			logger.log({"ah":ah, "rh":rh, "trh":trh, "lc":lc, "lp":lp, 
						"av_rw":av_rw, "min_rw":min_rw, "max_rw":max_rw, 
						"ph_path":phase_path, "s":entropy, "avq":avq,
						"crit_act_h": critic_act_h,
						"pol_act_h":policy_act_h}, episode)

			loss_critic = []
			loss_q = []
			rewards = []
			phase_path = []
			episode += 1

	env.close()
	ddpg.save_model(path='.')