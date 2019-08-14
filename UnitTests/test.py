import os
import sys
import unittest
import subprocess

import _GodotEnv

def launch_process(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        success = False
    return output, success

class TestExternalTest(unittest.TestCase):
    def setUp(self):
        output = launch_process(["g++", "external_program.cpp", "-lpthread", "-lrt"])
        print(output)
        # print(subprocess.check_output(["g++", "external_program.cpp", "-lpthread", "-lrt"], stderr=subprocess.STDOUT).decode())
        # os.system('g++ external_program.cpp -lpthread -lrt')
        self.mem = _GodotEnv.SharedMemory("test_segment")
    
    def runTest(self):
        handle_name = self.mem.getHandle()
        print("./a.out", "test_segment", str(handle_name))
        output = launch_process(["./a.out", "test_segment", str(handle_name)])
        print(output)


if __name__=='__main__':
    unittest.main()