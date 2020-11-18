"""
Reimplementation of Continuous lunar lander in Godot
"""

import math
import subprocess
import torch
import _GodotEnv
import numpy as np
import atexit
import os

import gym
from gym import spaces

class MountainCarEnv(gym.Env):
	"""
	Description:
		Continuous MountainCar environment
	Observation: 
		Type: Box(3)
		Num	Observation     
		0	Pos_y
		1	Vel_
		2   Rotation

		Type: [Binary(1), Binary(1)]
		Num	Action
		0	left
        1	right
		
		
	Starting State:
		Initialized with random position and velocity
	Episode Termination:
		Crashing, going outside of the screen or standing still
	"""

	def __init__(self, exec_path, env_path):
		
		self.steps = 0
		self.max_steps = 200
		
		self.min_position = 150
		self.max_position = 400
		self.max_speed = 450
		self.rotation = 2*np.pi
		self.low = np.array([self.min_position, -self.max_speed, -self.rotation], dtype=np.float32)
		self.high = np.array([self.max_position, self.max_speed, -self.rotation], dtype=np.float32)

		self.handle = "Environment"
		self.mem = _GodotEnv.SharedMemoryTensor(self.handle)
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)
		
		#Important: if this process is called with subprocess.PIPE, the semaphores will be stuck in impossible combination
		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			self.process = subprocess.Popen([exec_path, "--path", os.path.abspath(env_path), "--handle", self.handle], stdout=out, stderr=err)
				

		#Array to manipulate the state of the simulator
		self.env_action = torch.zeros(2, dtype=torch.int, device='cpu')
		self.env_action[0] = 0	#1 = reset
		self.env_action[1] = 0	#1 = exit

		#Example of agent action
		self.agent_action = torch.zeros(1, dtype=torch.float, device='cpu')
		
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

		atexit.register(self.close)

		

	def seed(self, seed=None):
		pass

	def step(self, action):
		action = torch.tensor([action *2 -1], device='cpu', dtype=torch.float)
		self.mem.sendFloat("agent_action", action)
		self.mem.sendInt("env_action", self.env_action)
		
		self.sem_act.post()
		
		self.sem_obs.wait()
		
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")
		
		if self.steps < self.max_steps:
			self.steps+=1
			return observation.numpy(), reward.item(), done.item(), None
		else:
			return observation.numpy(), reward.item(), 1, None
		
	def reset(self):
		
		env_action = torch.tensor([1, 0], device='cpu', dtype=torch.int)
		self.mem.sendFloat("agent_action", self.agent_action)
		self.mem.sendInt("env_action", env_action)
		self.sem_act.post()

		self.sem_obs.wait()
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")

		self.steps = 0
		return observation.numpy()
		

	def render(self, mode='human'):
		pass

	def close(self):
		self.process.terminate()
		print("Terminated")


if __name__=='__main__':
	GODOT_BIN_PATH = "MountainCar/MountainCar.x86_64"
	env_abs_path = "MountainCar/MountainCar.pck"
	env = MountainCarEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)
	for i in range(10):
		obs = env.reset()
		print(obs)
		done = 0
		
		while done == 0:
			s_prime, r, done, info = env.step(torch.randn(1, dtype=torch.float, device='cpu'))
			#print(s_prime, r, done)
			if done == 1:
				break
		
	env.close()
