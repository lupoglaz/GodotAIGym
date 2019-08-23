import os
import sys
import unittest
import subprocess
import time
import threading

import torch
import _GodotEnv

def output_reader(proc, file):
    while True:
        byte = proc.stdout.read(1)
        if byte:
            sys.stdout.buffer.write(byte)
            sys.stdout.flush()
            file.buffer.write(byte)
        else:
            break

def launch_process(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        success = False
    return output, success

class TestSendReceive(unittest.TestCase):
    def setUp(self):
        output = launch_process(["g++", "external_program.cpp", "-lpthread", "-lrt"])
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
    def setUp(self):
        output = launch_process(["g++", "semaphores.cpp", "-lpthread", "-lrt"])
        print(output)
        self.sem = _GodotEnv.SharedMemorySemaphore("test_semaphore", 0)
    
    def runTest(self):
        subprocess.Popen(["./a.out", "test_semaphore"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p = subprocess.Popen(['sleep', '5'])
        while p.poll() is None:
            self.sem.wait()
            print('Still sleeping')
            self.sem.post()
            

        print('Not sleeping any longer.  Exited with returncode %d' % p.returncode) 
        # with    subprocess.Popen(['python', 'python_counter.py', '0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc1, \
        #         subprocess.Popen(["./a.out", "test_semaphore"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc2, \
        #         open('log1.log', 'w') as file1, \
        #         open('log2.log', 'w') as file2:
        # with    subprocess.Popen(['./python_counter.py', '0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc1, \
        #         subprocess.Popen(['./python_counter.py', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc2, \
        #         open('log1.log', 'w') as file1, \
        #         open('log2.log', 'w') as file2:
        #     t1 = threading.Thread(target=output_reader, args=(proc1, file1))
        #     t2 = threading.Thread(target=output_reader, args=(proc2, file2))
        #     t1.start()
        #     t2.start()
        #     t1.join()
        #     t2.join()


if __name__=='__main__':
    unittest.main()