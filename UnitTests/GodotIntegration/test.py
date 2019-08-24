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
		self.sem = _GodotEnv.SharedMemorySemaphore("sem3", 0)
		# print(launch_process(["./TestEnvironment.x86_64", "--handle", "environment"]))
	
	def runTest(self):
		action = torch.ones(4, dtype=torch.int, device='cpu')
		action[0] = 0 
		action[1] = 1
		action[2] = 0
		action[3] = 1
		
		self.mem.send("action", action)
		p = subprocess.Popen(["./TestEnvironment.x86_64", "--handle", "environment"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
		time.sleep(1.0)
		self.sem.post()

		while p.poll() is None:
			print("Waiting")
			time.sleep(0.1)
			# self.sem.post()
			# time.sleep(0.1)
			# self.sem.wait()
			# observation = self.mem.receive("observation")
			# print(observation)

if __name__=='__main__':
	unittest.main()