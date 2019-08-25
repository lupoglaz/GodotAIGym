import os
import sys
import unittest
import subprocess
import time
import threading

import torch
import _GodotEnv

def launch_process(command):
	try:
		output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
		success = True
	except subprocess.CalledProcessError as e:
		output = e.output.decode()
		success = False
	return output, success

class TestGodotEnvironment(unittest.TestCase):
	"""
	Compiles program tensors.cpp, which reads a tensor from shared memory and wites another one back
	"""
	def setUp(self):
		self.mem = _GodotEnv.SharedMemoryTensor("environment")
		self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
		self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)
			
	def runTest(self):
		action = torch.ones(4, dtype=torch.int, device='cpu')
		action[0] = 0 
		action[1] = 1
		action[2] = 0
		action[3] = 1
		with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
			p = subprocess.Popen(["./TestEnvironment.x86_64", "--handle", "environment"], stdout=out, stderr=err)
		
		while p.poll() is None:
			action = torch.randint(0, 2, (4,), dtype=torch.int, device='cpu')
			self.mem.send("action", action)
			self.sem_act.post()
			print("Sent action", action)

			# print("Waiting")
			# time.sleep(0.1)
			
			self.sem_obs.wait()
			observation = self.mem.receive("observation")
			print("Received observation", observation)
		
if __name__=='__main__':
	unittest.main()