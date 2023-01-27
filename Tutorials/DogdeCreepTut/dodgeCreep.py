import math

import subprocess
import torch
import tensorflow as tf
import _GodotEnv
import numpy as np
import atexit
import os
import time

import gym
from gym import spaces


class dodgeCreepEnv(gym.Env):
	def __init__(self, exec_path, env_path, num_actions=5, num_observations=128*128*3, turbo_mode=False):
		self.handle = "environment"
		self.mem = _GodotEnv.SharedMemoryTensor(self.handle)
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)

		self.agent_action_tensor = self.mem.newFloatTensor("agent_action", 1)
		self.env_action_tensor = self.mem.newIntTensor("env_action", 1)
		self.observation_tensor = self.mem.newUintTensor("observation", num_observations)
		self.reward_tensor = self.mem.newFloatTensor("reward", 1)
		self.done_tensor = self.mem.newIntTensor("done", 1)

		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			if turbo_mode:
				#exec_path = exec_path + " t"
				self.process = subprocess.Popen([exec_path, "t" ,"--path", os.path.abspath(env_path), "--handle", self.handle, "--fixed-fps 600"], stdout=out, stderr=err)
			else:
				self.process = subprocess.Popen([exec_path, "n" ,"--path", os.path.abspath(env_path), "--handle", self.handle], stdout=out, stderr=err)

		#Array to manipulate the state of the simulator
		self.env_action = torch.zeros(2, dtype=torch.int, device='cpu')
		self.env_action[0] = 0	#1 = reset
		self.env_action[1] = 0	#1 = exit

		#Example of agent action
		self.agent_action = torch.zeros(1, dtype=torch.float, device='cpu')

		self.max_speed = 8.0

		self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(num_actions,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_observations,), dtype=np.float32)

		atexit.register(self.close)

	def seed(self, seed=None):
		pass

	def step(self, action):
		action = np.array([action])
		action = torch.from_numpy(action)
		self.env_action_tensor.write(self.env_action)
		self.agent_action_tensor.write(action.to(dtype=torch.float32))
		self.sem_act.post()

		self.sem_obs.wait()
		observation = self.observation_tensor.read()
		reward = self.reward_tensor.read()
		done = self.done_tensor.read()

		return observation.detach().numpy(), reward.detach().numpy(), done.detach().numpy(), None
	
	def reset(self, seed=42):
		action = np.array([0])
		action = torch.from_numpy(action)
		env_action = torch.tensor([1, 0], device='cpu', dtype=torch.int)
		self.env_action_tensor.write(env_action.to(dtype=torch.int32))
		self.agent_action_tensor.write(action.to(dtype=torch.float32))
		self.sem_act.post()

		self.sem_obs.wait()
		observation = self.observation_tensor.read()
		reward = self.reward_tensor.read()

		return observation.detach().numpy()
	
	def render(self, mode='human'):
		pass

	def close(self):
		self.process.terminate()
		print("Terminated")


if __name__=='__main__':
	GODOT_BIN_PATH = "DodgeCreep/DodgeCreep.x86_64"
	env_abs_path = "DodgeCreep/DodgeCreep.pck"
	env_my = InvPendulumEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)
	for i in range(1000):
		obs_my, rew_my, done, _ = env_my.step(torch.tensor([8.0]))
		print(rew_my, obs_my, done)

	env_my.close()
	sys.exit()
	# env_my.reset()

	gym_obs = []
	#gym_rew = []
	my_obs = []
	#my_rew = []
	for i in range(1000):
		obs_my, rew_my, done, _ = env_my.step(torch.tensor([8.0]))
		obs, rew, done, _ = env.step(np.array([2.0]))
		env.render()
		gym_obs.append(obs)
		gym_rew.append(rew)
		my_obs.append(obs_my)
		my_rew.append(rew_my)
	
	env_my.close()
	
	gym_obs = np.array(gym_obs)
	gym_rew = np.array(gym_rew)
	my_obs = torch.stack(my_obs, dim=0).numpy()
	my_rew = np.array(my_rew)

	'''
	plt.subplot(1,4,1)
	plt.plot(gym_rew, label='Gym rewards')
	plt.plot(my_rew, label='My rewards')
	plt.subplot(1,4,2)
	plt.plot(gym_obs[:,0], label='gym obs0')
	plt.plot(my_obs[:,0], label='my obs0')
	plt.subplot(1,4,3)
	plt.plot(gym_obs[:,1], label='gym obs1')
	plt.plot(my_obs[:,1], label='my obs1')
	plt.subplot(1,4,4)
	plt.plot(gym_obs[:,2], label='gym obs2')
	plt.plot(my_obs[:,2], label='my obs2')
	plt.legend()
	plt.show()
	'''