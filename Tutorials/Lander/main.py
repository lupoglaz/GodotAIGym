import torch
import gym
from stable_baselines3 import DDPG
from LanderEnv import LanderEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

class NormalizedEnvironment(gym.ActionWrapper):
    def action(self, action):
        act_k = torch.from_numpy(self.action_space.high - self.action_space.low).to(dtype=torch.float32)/2.0
        act_b = torch.from_numpy(self.action_space.high + self.action_space.low).to(dtype=torch.float32)/2.0
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2.0/(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/2.0
        return act_k_inv * (action - act_b)

if __name__ == '__main__':
    train = True
    trace = True
    env = NormalizedEnvironment(LanderEnv(exec_path='LanderEnv/LanderEnv.x86_64', pck_path='LanderEnv/LanderEnv.pck', render=True))
    if train:
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
        model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log='./Log/')
        model.learn(total_timesteps=500_000)
        model.save("ddpg")
    if trace:
        model = DDPG.load("ddpg")
        observation = env.reset()
        observation = torch.from_numpy(observation).to(dtype=torch.float32)
        actor = model.policy.actor.mu.to(device='cpu', dtype=torch.float32)
        traced_actor = torch.jit.trace(actor, observation)
        traced_actor.save('ddpg_actor.jit')
    env.close()
