import torch
import torch.nn as nn
import numpy as np

from Model import Actor, Critic
from torch.distributions.uniform import Uniform
class DDPG:
	def __init__(self, env, tau=1e-3, gamma=0.99, batch_size=64, depsilon=50000):
		self.num_states = env.observation_space.shape[0]
		self.num_actions = env.action_space.shape[0]
		
		self.policy = Actor(self.num_states, self.num_actions).train()
		self.policy_target = Actor(self.num_states, self.num_actions).eval()
		self.hard_update(self.policy, self.policy_target)
		
		self.critic = Critic(self.num_states, self.num_actions).train()
		self.critic_target = Critic(self.num_states, self.num_actions).eval()
		self.hard_update(self.critic, self.critic_target)

		self.critic_loss = nn.MSELoss()

		self.batch_size = batch_size
		self.gamma = gamma
		self.tau = tau
		self.epsilon = 1.0
		self.depsilon = 1.0/float(depsilon)
		
		self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=1e-3)
		self.opt_policy = torch.optim.Adam(self.policy.parameters(),lr=1e-4)
		
		self.policy.cuda()
		self.policy_target.cuda()
		self.critic.cuda()
		self.critic_target.cuda()
		
	def train(self, buffer):
		b_state, b_action, b_reward, b_state_next, b_term = buffer.sample(self.batch_size)
		with torch.no_grad():
			action_target = self.policy_target(b_state_next)
			Q_prime = self.critic_target(b_state_next, action_target)

		self.opt_critic.zero_grad()
		Q = self.critic(b_state, b_action)
		L_critic = self.critic_loss(Q, b_reward + self.gamma*Q_prime*(1.0-b_term))
		L_critic.backward()
		self.opt_critic.step()
		
		self.opt_policy.zero_grad()
		action = self.policy(b_state)
		L_Q = -1.0*self.critic(b_state, action).mean()
		L_Q.backward()
		self.opt_policy.step()

		self.soft_update(self.critic, self.critic_target)
		self.soft_update(self.policy, self.policy_target)

		return L_critic.item(), L_Q.item()

	def get_entropy(self, buffer, m=5, n=100):
		# b_state, b_action, b_reward, b_state_next, b_term = buffer.sample(n)
		b_angle = torch.rand(n)*np.pi*2.0
		b_speed = 2.0*(torch.rand(n)-0.5)*8.0
		b_state = torch.stack([torch.cos(b_angle), torch.sin(b_angle), b_speed], dim=1).to(device='cuda', dtype=torch.float32)
		coef = torch.zeros(n, dtype=b_state.dtype, device=b_state.device)
		with torch.no_grad():
			action = self.policy(b_state)
			X, ind = torch.sort(action, dim=0)
			for i in range(n):
				if i < m:
					c = 1
					a = X[i+m]
					b = X[0]
				elif i >= m and i < n-m:
					c = 2
					a = X[i+m]
					b = X[i-m]
				else:
					c = 1
					a = X[n-1]
					b = X[i-m]
				coef[i] = float(n)*float(c)/float(m)*(a - b + 1E-5)

			S = torch.log(coef).mean()
		
		return S.item()
	
	def get_value(self, state, action):
		with torch.no_grad():
			return self.critic(state, action).item()

	def select_action(self, state, random_process):
		with torch.no_grad():
			action = self.policy(state)
		noise = max(self.epsilon, 0.0)*random_process.sample()
		self.epsilon -= self.depsilon

		action += torch.from_numpy(noise).to(device=action.device, dtype=action.dtype)
		action = torch.clamp(action, -1, 1)
		return action

	def random_action(self):
		m = Uniform(torch.tensor([-1.0 for i in range(self.num_actions)]), 
					torch.tensor([1.0 for i in range(self.num_actions)]))
		return m.sample()

	def soft_update(self, src, dst):
		with torch.no_grad():
			for src_param, dst_param in zip(src.parameters(), dst.parameters()):
				dst_param.copy_(self.tau * src_param + (1.0 - self.tau) * dst_param)

	def hard_update(self, src, dst):
		with torch.no_grad():
			for src_param, dst_param in zip(src.parameters(), dst.parameters()):
				dst_param.copy_(src_param.clone())

	def load_weights(self, path):
		self.policy.load_state_dict(torch.load('{}/policy.pkl'.format(path)))
		self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))

	def save_model(self, path):
		torch.save(self.policy.to(device='cpu').state_dict(), '{}/policy.pkl'.format(path))
		torch.save(self.critic.to(device='cpu').state_dict(), '{}/critic.pkl'.format(path))