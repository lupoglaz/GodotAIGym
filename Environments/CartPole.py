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

class CartPoleEnv(gym.Env):
	"""
	Description:
		A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
	Source:
		This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
	Observation: 
		Type: Box(4)
		Num	Observation                 Min         Max
		0	Cart Position             -4.8            4.8
		1	Cart Velocity             -Inf            Inf
		2	Pole Angle                 -24 deg        24 deg
		3	Pole Velocity At Tip      -Inf            Inf
		
	Actions:
		Type: Discrete(2)
		Num	Action
		0	Push cart to the left
		1	Push cart to the right
		
		Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
	Reward:
		Reward is 1 for every step taken, including the termination step
	Starting State:
		All observations are assigned a uniform random value in [-0.05..0.05]
	Episode Termination:
		Pole Angle is more than 12 degrees
		Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
		Episode length is greater than 200
		Solved Requirements
		Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
	"""

	def __init__(self):
		self.mem = _GodotEnv.SharedMemoryTensor("environment")
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)

		#Important: if this process is called with subprocess.PIPE, the semaphores will be stuck in impossible combination
		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			p = subprocess.Popen(["./CartPole.x86_64", "--handle", "environment"], stdout=out, stderr=err)
		
		#Array to manipulate the state of the simulator
		self.env_action = torch.zeros(2, dtype=torch.int, device='cpu')
		self.env_action[0] = 0	#1 = reset
		self.env_action[1] = 0	#1 = exit

		#Example of agent action (0/1)
		self.agent_action = torch.zeros(1, dtype=torch.int, device='cpu')

	def seed(self, seed=None):
		pass

	def step(self, action):
		action = torch.tensor([1], dtype=torch.int, device='cpu')
		self.mem.sendInt("agent_action", action)
		self.mem.sendInt("env_action", self.env_action)
		self.sem_act.post()
		
		self.sem_obs.wait()
		observation = self.mem.receiveFloat("observation")
		reward = self.mem.receiveFloat("reward")
		done = self.mem.receiveInt("done")

		return observation, reward, done.item(), None
		
	def reset(self):
		env_action = torch.tensor([1, 0], device='cpu', dtype=torch.int)
		self.mem.sendInt("agent_action", self.agent_action)
		self.mem.sendInt("env_action", env_action)
		self.sem_act.post()
		

	def render(self, mode='human'):
		pass

	def close(self):
		env_action = torch.tensor([0, 1], device='cpu', dtype=torch.int)
		self.mem.sendInt("agent_action", self.agent_action)
		self.mem.sendInt("env_action", env_action)
		self.sem_act.post()

if __name__=='__main__':
	env = CartPoleEnv()
	for i in range(10):
		done = False
		# s = env.reset()
		while not done:
			for t in range(100):
				s_prime, r, done, info = env.step(np.array([1]))
				if done:
					break
	env.close()