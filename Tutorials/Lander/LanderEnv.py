import os
import torch
import _GodotEnv
import subprocess
import atexit
import gym
from gym import spaces
import numpy as np

class LanderEnv(gym.Env):
    def __init__(self, exec_path, pck_path=None, render=False, num_actions=2, num_observations=21):
        self.handle = 'environment'
        self.mem = _GodotEnv.SharedMemoryTensor(self.handle)
        self.sem_act = _GodotEnv.SharedMemorySemaphore("sem_action", 0)
        self.sem_obs = _GodotEnv.SharedMemorySemaphore("sem_observation", 0)

        self.agent_action_tensor = self.mem.newFloatTensor("agent_action", num_actions)
        self.env_action_tensor = self.mem.newIntTensor("env_action", 2)
        self.observation_tensor = self.mem.newFloatTensor("observation", num_observations)
        self.reward_tensor = self.mem.newFloatTensor("reward", 1)
        self.done_tensor = self.mem.newIntTensor("done", 1)

        self.out = open("stdout.txt", 'wb')
        self.err = open("stderr.txt", 'wb')

        cmd = [exec_path]
        if not(pck_path is None):
            cmd += ['--path', os.path.abspath(pck_path)]
        if not render:
            cmd += ['--disable-render-loop']
        cmd += ['--handle', self.handle]
        self.process = subprocess.Popen(cmd, stdout=self.out, stderr=self.err)

        self.action_space = spaces.Box(low=-1, high=1.0, shape=(num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1.0, shape=(num_observations,), dtype=np.float32)

        atexit.register(self.close)
    
    def render(self, mode='human'):
        pass

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        self.env_action_tensor.write(torch.tensor([0, 0], dtype=torch.int32, device='cpu'))
        self.agent_action_tensor.write(action.to(dtype=torch.float32))
        self.sem_act.post()

        self.sem_obs.wait()
        observation = self.observation_tensor.read()
        reward = self.reward_tensor.read()
        done = self.done_tensor.read()
        return observation.numpy(), reward.item(), done.item(), {}

    def reset(self, seed=42):
        self.env_action_tensor.write(torch.tensor([1, 0], dtype=torch.int32, device='cpu'))
        self.sem_act.post()

        self.sem_obs.wait()
        observation = self.observation_tensor.read()
        return observation.numpy()

    def close(self):
        self.env_action_tensor.write(torch.tensor([0, 1], dtype=torch.int32, device='cpu'))
        self.sem_act.post()

        self.sem_obs.timed_wait(1000)
                
        self.out.close()
        self.err.close()

if __name__ == '__main__':
    env = LanderEnv(exec_path='LanderEnv/LanderEnv.x86_64', pck_path='LanderEnv/LanderEnv.pck', render=True)
    for i in range(1000):
        obs, rew, done, info = env.step(torch.randn(2, dtype=torch.float32, device='cpu'))
        if done:
            env.reset()
    env.close()
    