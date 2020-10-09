import gym
import torch

class NormalizedEnv(gym.ActionWrapper):
	def action(self, action):
		# act_k = (self.action_space.high - self.action_space.low)/2.0
		# act_b = (self.action_space.high + self.action_space.low)/2.0
		act_k = torch.from_numpy(self.action_space.high - self.action_space.low)/2.0
		act_b = torch.from_numpy(self.action_space.high + self.action_space.low)/2.0
		return act_k * action + act_b

	def reverse_action(self, action):
		act_k_inv = 2.0/(self.action_space.high - self.action_space.low)
		act_b = (self.action_space.high + self.action_space.low)/2.0
		return act_k_inv * (action - act_b)