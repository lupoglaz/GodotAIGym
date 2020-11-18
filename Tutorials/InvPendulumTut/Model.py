import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.uniform_(m.bias.data)

def fanin_init(m):
	if isinstance(m, nn.Linear):
		fanin = m.weight.data.size(0)
		v = 1. / np.sqrt(fanin)
		m.weight.data.uniform_(-v, v)

class Actor(nn.Module):
	def __init__(self, num_states, num_actions, hidden=[64, 64]):
		super(Actor, self).__init__()
		
		self.fc1 = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1]),
			nn.ReLU()
		)
		self.fc3 = nn.Sequential(
			nn.Linear(hidden[1], num_actions),
			nn.Tanh()
		)
		self.apply(fanin_init)
		self.fc3[0].weight.data.uniform_(-3e-3, 3e-3)
	
	def get_act_hist(self):
		with torch.no_grad():
			data = torch.cat([self.out1.flatten(), self.out2.flatten(), self.out3.flatten()], dim=0)
		return np.histogram(data.cpu().numpy(), bins=60)

	def forward(self, x):
		self.out1 = self.fc1(x)
		self.out2 = self.fc2(self.out1)
		self.out3 = self.fc3(self.out2)
		return self.out3

class Critic(nn.Module):
	def __init__(self, num_states, num_actions, hidden=[64, 64]):
		super(Critic, self).__init__()
		self.state_net = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU()
		)
		self.state_action_net = nn.Sequential(
			nn.Linear(hidden[0] + num_actions, hidden[1]),
			nn.ReLU(),
		)
		self.fc_fin = nn.Sequential(
			nn.Linear(hidden[1], 1)
		)
		self.fc_fin[0].weight.data.uniform_(-3e-3, 3e-3)
		self.apply(fanin_init)
	
	def get_act_hist(self):
		with torch.no_grad():
			data = torch.cat([self.out1.flatten(), self.out2.flatten(), self.out3.flatten()], dim=0)
		return np.histogram(data.cpu().numpy(), bins=60)

	def forward(self, state, action):
		self.out1 = self.state_net(state)
		state_action = torch.cat([self.out1, action], dim=-1)
		self.out2 = self.state_action_net(state_action)
		self.out3 = self.fc_fin(self.out2).squeeze()
		return self.out3