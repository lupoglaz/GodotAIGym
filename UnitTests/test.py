import os
import sys
import unittest
import subprocess
import time
import threading

import torch
import _GodotEnv
import asyncio


def launch_process(command):
	try:
		output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
		success = True
	except subprocess.CalledProcessError as e:
		output = e.output.decode()
		success = False
	return output, success

class TestSendReceive(unittest.TestCase):
	"""
	Compiles program tensors.cpp, which reads a tensor from shared memory and wites another one back
	"""
	def setUp(self):
		output = launch_process(["g++", "tensors.cpp", "-lpthread", "-lrt"])
		print(output)
		self.mem = _GodotEnv.SharedMemoryTensor("test_segment")
	
	def runTest(self):
		T_out = torch.ones(10, dtype=torch.int, device='cpu')
		print("Python to c++:", T_out)
		self.mem.send("T_out", T_out);
		output = launch_process(["./a.out", "test_segment", "T_out", "T_in"])
		print(output)
		
		T_in = self.mem.receive("T_in");
		print("C++ to Python:", T_in)


class TestSemaphore(unittest.TestCase):
	"""
	Compiles program semaphores.cpp, which waits for a semaphore and writes a number to afile.dat
	This test increments semaphore onece 0.1 s and reads from file

	If semaphore does not work, we won't read consequtive numbers, otherwise test successfull
	"""
	def setUp(self):
		output = launch_process(["g++", "semaphores.cpp", "-lpthread", "-lrt"])
		print(output)
		self.sem = _GodotEnv.SharedMemorySemaphore("test_semaphore", 0)
	
	def runTest(self):
		p = subprocess.Popen(["./a.out", "test_semaphore"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		while p.poll() is None:
			self.sem.post()
			time.sleep(0.1)
			with open("afile.dat", 'r') as fin:
				print("Read from file = ", fin.readline())


if __name__=='__main__':
	unittest.main()