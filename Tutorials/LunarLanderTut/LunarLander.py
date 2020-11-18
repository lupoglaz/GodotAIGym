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

class LunarLanderEnv(gym.Env):
	"""
	Description:
		Continuous LunarLander environment
	Observation: 
		Type: Box(7)
		Num	Observation     
		0	Pos_x
		1	Pos_y
		2	Vel_x
        3	Vel_y
        4   Angle
        5   Angular_vel
        6   Left leg touch
        7   Right leg touch
		
	Actions:
		Type: [Continuous(1), Continuous(1)]
		Num	Action
		0	Main Engine
        1	Side Engines
		
	Reward:
		
	Starting State:
		Initialized with random position and velocity
	Episode Termination:
		Crashing, going outside of the screen or standing still
	"""

	def __init__(self, exec_path, env_path):
		
		self.handle = "environment"
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
		self.agent_action = torch.zeros(2, dtype=torch.float, device='cpu')
		
		high = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

		atexit.register(self.close)

	def seed(self, seed=None):
		pass

	def step(self, action):
		self.mem.sendFloat("agent_action", action)
		self.mem.sendInt("env_action", self.env_action)
		self.sem_act.post()
		
		self.sem_obs.wait()
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")

		return observation, reward.item(), done.item(), None
		
	def reset(self):
		env_action = torch.tensor([1, 0], device='cpu', dtype=torch.int)
		self.mem.sendFloat("agent_action", self.agent_action)
		self.mem.sendInt("env_action", env_action)
		self.sem_act.post()

		self.sem_obs.wait()
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")
		return observation
		

	def render(self, mode='human'):
		pass

	def close(self):
		self.process.terminate()
		print("Terminated")


if __name__=='__main__':
	GODOT_BIN_PATH = "LunarLander/LunarLander.x86_64"
	env_abs_path = "LunarLander/LunarLander.pck"
	env = LunarLanderEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)
	for i in range(10):
		obs = env.reset()
		print(obs)
		done = 0
		while done == 0:
			s_prime, r, done, info = env.step(torch.randn(2, dtype=torch.float, device='cpu'))
			print(s_prime, r, done)
			if done == 1:
				break
		
	env.close()