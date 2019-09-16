"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import subprocess
import torch
import _GodotEnv
import numpy as np
import atexit
import os

class InvPendulumEnv(gym.Env):
	"""
	Description:
		Standard inverted pendulum environment but for the discrete action space
	Observation: 
		Type: Box(3)
		Num	Observation     
		0	Cos(theta)      
		1	Sin(theta)      
		2	Angular velocity
		
	Actions:
		Type: [Discrete(2), Discrete(2)]
		Num	Action
		0	Apply positive torque
		1	Apply negative torque
		
	Reward:
		var n_theta = fmod((theta + PI), (2*PI)) - PI
		reward = (n_theta*n_theta + .1*angular_velocity*angular_velocity)
	Starting State:
		Initialized with random angle
	Episode Termination:
		Terminated after 10s
	"""

	def __init__(self, exec_path, env_path):
		self.mem = _GodotEnv.SharedMemoryTensor("environment")
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)

		#Important: if this process is called with subprocess.PIPE, the semaphores will be stuck in impossible combination
		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			self.process = subprocess.Popen([exec_path, "--handle", "environment", "--path", os.path.abspath(env_path)], stdout=out, stderr=err)
		
		#Array to manipulate the state of the simulator
		self.env_action = torch.zeros(2, dtype=torch.int, device='cpu')
		self.env_action[0] = 0	#1 = reset
		self.env_action[1] = 0	#1 = exit

		#Example of agent action [(0/1), (0/1)]
		self.agent_action = torch.zeros(2, dtype=torch.int, device='cpu')

		atexit.register(self.close)

	def seed(self, seed=None):
		pass

	def step(self, action):
		# print("Sending action")
		self.mem.sendInt("agent_action", action)
		self.mem.sendInt("env_action", self.env_action)
		self.sem_act.post()
		# print("Waiting for observation")
		
		self.sem_obs.wait()
		# print("Reading observation")
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")
		# print("Read observation")

		return observation, reward.item(), done.item(), None
		
	def reset(self):
		# print("Sending reset action")
		env_action = torch.tensor([1, 0], device='cpu', dtype=torch.int)
		self.mem.sendInt("agent_action", self.agent_action)
		self.mem.sendInt("env_action", env_action)
		self.sem_act.post()
		# print("Sent reset action")

		self.sem_obs.wait()
		# print("Reading observation")
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")
		# print("Read observation")
		return observation
		

	def render(self, mode='human'):
		pass

	def close(self):
		self.process.terminate()
		print("Terminated")


if __name__=='__main__':
	GODOT_BIN_PATH = "InvPendulum/InvPendulum.x86_64"
	env_abs_path = "InvPendulum/InvPendulum.pck"
	env = InvPendulumEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)
	for i in range(10):
		done = 0
		while done == 0:
			s_prime, r, done, info = env.step(torch.tensor([0, 1], dtype=torch.int, device='cpu'))
			print(s_prime, r, done)
			if done == 1:
				break
		env.reset()
	env.close()