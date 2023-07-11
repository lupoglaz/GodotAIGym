import torch
import gym
import _GodotEnv
import subprocess
import atexit


class DummyEnv(gym.Env):
    def __init__(self, env_path, num_actions, num_observations):
        self.handle = "environment"
        self.mem = _GodotEnv.SharedMemory("environment")
        # self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
        # self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)
        # self.sem_env = _GodotEnv.SharedMemorySemaphore("sem_environment", 0)
        
        #Shared memory tensors
        # self.agent_action_tensor = self.mem.newFloatTensor("agent_action", num_actions)
        # self.observation_tensor = self.mem.newFloatTensor("observation", num_observations)
        
        with open("stdout.txt","wb") as out, open("stderr.txt","wb") as err:
            self.process = subprocess.Popen([env_path], stdout=out, stderr=err)

        # self.sem_env.wait()
        atexit.register(self.close)
    
    def close(self):
        self.process.terminate()

    def seed(self, seed=None):
        pass

    def step(self, action):
        print('Writing action')
        # self.agent_action_tensor.write(action)
        # self.sem_act.wait()

        # print('Waiting observation')
        # self.sem_obs.wait()
        #receiving observation
        # observation = self.observation_tensor.read()
        
        observation, reward, done,info = None, None, None, None
        return observation, reward, done, info
        
    def reset(self):
        observation = None
        return observation
        
    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__=='__main__':
    env = DummyEnv('./SharedMemoryTest.x86_64', 1, 1)
    act = torch.tensor([8.0])
    for i in range(100):
        print(f'Action step {i} : {act}')
        obs, _, _, _ = env.step(act)
        print(f'Observation step {i} : {obs}')