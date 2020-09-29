"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import subprocess
import torch
import _GodotEnv
import atexit
import os

import gym
from gym import spaces

import time
import json

class InvPendulumEnv(gym.Env):
	"""
	
	"""

	def __init__(self, exec_path, env_path):
		self.handle = "environment"
		self.mem = _GodotEnv.SharedMemoryTensor(self.handle)
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)

		#Important: if this process is called with subprocess.PIPE, the semaphores will be stuck in impossible combination
		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			self.process = subprocess.Popen([exec_path, "--path", os.path.abspath(env_path),"--disable-render-loop","--handle", self.handle], stdout=out, stderr=err)
		time.sleep(0.01)
		#Array to manipulate the state of the simulator
		# self.env_action = torch.zeros(1, dtype=torch.int, device='cpu')
		# self.env_action[0] = 0	#1 = exit	
		
		atexit.register(self.close)

	def seed(self, seed=None):
		pass

	def step(self):
		# self.mem.sendInt("env_action", self.env_action)
		self.sem_act.post()
				
		self.sem_obs.wait()
		observation = self.mem.receiveFloat("observation")
				
		return observation

	def render(self, mode='human'):
		pass

	def close(self):
		self.process.terminate()
		print("Terminated")


if __name__=='__main__':
	from matplotlib import pylab as plt
	
	GODOT_BIN_PATH = "IntCheck/IntCheck.x86_64"
	env_abs_path = "IntCheck/IntCheck.pck"
	env_my = InvPendulumEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path)

	my_obs = []
	for i in range(20):
		obs_my = env_my.step()
		print(obs_my)
		if (i+1) == 10:
			time.sleep(2.0)
		my_obs.append(obs_my)
	
	env_my.close()

	with open("log.json", 'rt') as fin:
		dict = json.load(fin)
	
	my_obs = torch.stack(my_obs, dim=0)
	f = plt.figure(figsize=(20,3))
	plt.subplot(1,5,1)
	plt.plot(my_obs[:,5], my_obs[:,0], label='Coord/time')
	plt.legend()
	plt.subplot(1,5,2)
	plt.plot(my_obs[:,1], label='Iter/sec')
	plt.legend()
	plt.subplot(1,5,3)
	plt.plot(my_obs[:,2], label='Frames/sec')
	plt.legend()
	plt.subplot(1,5,4)
	plt.plot(my_obs[:,3], label='Time scale')
	plt.legend()
	plt.subplot(1,5,5)
	plt.plot(my_obs[:,4], label='Frame time')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	f = plt.figure(figsize=(20,3))
	plt.subplot(1,3,1)
	plt.plot(dict["t"], dict["Kin"], label='Real time')
	plt.legend()
	plt.subplot(1,3,2)
	plt.plot(dict["t"], dict["Its"], label='Iterations/sec')
	plt.legend()
	plt.subplot(1,3,3)
	plt.plot(dict["t"], dict["Fps"], label='Frames/sec')
	plt.legend()
	
	plt.tight_layout()
	plt.show()

	