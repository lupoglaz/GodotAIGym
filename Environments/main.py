#################
#Code modified from https://github.com/nikhilbarhate99/PPO-PyTorch
#################


import os
import sys
import torch
import gym
import numpy as np
from PPO import PPO
from Memory import Memory
from LunarLander import LunarLanderEnv

import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='PPO+Godot')	
	parser.add_argument('-environment', default='LunarLanderContinuous-v2', help='Environment name')
	parser.add_argument('-test', default=None, help='Test only', type=str)
	parser.add_argument('-solved_reward', default=300.0, help='Reward, when the problem is solved', type=float)
	parser.add_argument('-log_interval', default=20, help='Log interval', type=int)
	parser.add_argument('-max_episodes', default=10000, help='Maximum number of episodes', type=int)
	parser.add_argument('-max_timesteps', default=500, help='Maximum number of timesteps in an episode', type=int)
	parser.add_argument('-update_timestep', default=4000 , help='Update period', type=int)

	parser.add_argument('-lr', default=0.0003 , help='Learning rate', type=float)
	parser.add_argument('-action_std', default=0.5, help='Max epoch', type=float)
	parser.add_argument('-K_epochs', default=80 , help='Number of policy updates', type=int)
	parser.add_argument('-eps_clip', default=0.2, help='PPO clipping parameter', type=float)
	parser.add_argument('-gamma', default=0.99, help='Reward discount', type=float)
	
	args = parser.parse_args()

	betas = (0.9, 0.999)
		
	# creating environment
	# env = gym.make(args.environment)
	GODOT_BIN_PATH = "LunarLander/LunarLander.x86_64"
	env_abs_path = "LunarLander/LunarLander.pck"
	env = LunarLanderEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	
	memory = Memory()
	ppo = PPO(state_dim, action_dim, args.action_std, args.lr, betas, args.gamma, args.K_epochs, args.eps_clip)
	
	if not args.test is None:
		filename = "PPO_continuous_solved_" +args.environment+ ".pth"
		ppo.policy_old.load_state_dict(torch.load(filename))
	
	# logging variables
	running_reward = 0
	avg_length = 0
	time_step = 0
	
	# training loop
	for i_episode in range(1, args.max_episodes+1):
		state = env.reset()
		for t in range(args.max_timesteps):
			time_step +=1
			# Running policy_old:
			# action = ppo.select_action(state, memory).cpu().data.numpy().flatten()
			action = ppo.select_action(state, memory).cpu().flatten()
			state, reward, done, _ = env.step(action)
			
			#Training
			if args.test is None: 
				# Saving reward and is_terminals:
				memory.rewards.append(reward)
				memory.is_terminals.append(done)
				
				# update if its time
				if time_step % args.update_timestep == 0:
					ppo.update(memory)
					memory.clear_memory()
					time_step = 0
			
			#Testing
			else:	
				env.render()
			running_reward += reward
			if done:
				break
				
		avg_length += t
		
		# stop training if avg_reward > solved_reward
		if (running_reward > (args.log_interval * args.solved_reward)) and args.test is None:
			print("########## Solved! ##########")
			torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(args.environment))
			break
		
		# save every 500 episodes
		if (i_episode % 500 == 0) and (args.test is None):
			torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(args.environment))
			
		# logging
		if i_episode % args.log_interval == 0:
			avg_length = int(avg_length/args.log_interval)
			running_reward = int((running_reward/args.log_interval))
			
			print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
			running_reward = 0
			avg_length = 0