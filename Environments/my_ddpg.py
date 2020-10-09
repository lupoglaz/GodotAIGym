import gym
import numpy as np
import random
import matplotlib.pylab as plt

import torch
import torch.nn as nn

class ReplayBuffer(object):
	def __init__(self, max_length=50000):
		self.max_length = max_length
		self.buffer = []

	def append(self, state, action, reward, state_next, term):
		self.buffer.append((state.clone(), action.detach().clone(), reward, state_next.clone(), term))
		if len(self.buffer)>self.max_length:
			self.buffer.pop(0)

	def sample(self, batch_size):
		state, action, reward, state_next, term = zip(*random.choices(self.buffer, k=batch_size))
		state = torch.stack(state, dim=0)
		state_next = torch.stack(state_next, dim=0)
		action = torch.stack(action, dim=0)
		reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
		term = torch.tensor(term, dtype=state.dtype, device=state.device)
		return state, action, reward, state_next, term
	
	def __len__(self):
		return len(self.buffer)
			
class Noise(object):
	def __init__(self, env_action, mu=0.0, theta=0.15, max_sigma=0.3,
				min_sigma=0.1, decay_period=5000):
		self.mu = mu
		self.theta = theta
		self.sigma = max_sigma
		self.max_sigma = max_sigma 
		self.min_sigma = min_sigma 
		self.decay_period = decay_period
		self.num_actions = env_action.shape[0]
		self.action_low = env_action.low[0]
		self.action_high = env_action.high[0]
		self.reset()

	def reset(self):
		self.state = np.zeros(self.num_actions)
	
	def state_update(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)
	
	def add_noise(self, action, training_step):
		self.state_update()
		state = torch.from_numpy(self.state).to(dtype=action.dtype, device=action.device)
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, training_step)/self.decay_period
		return torch.clamp(action + state, self.action_low, self.action_high)

class Actor(nn.Module):
	def __init__(self, num_states, num_actions, env_action, hidden=[400, 300]):
		super(Actor, self).__init__()
		self.env_action = env_action
		self.net = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU(),
			nn.Linear(hidden[0], hidden[1]),
			nn.ReLU(),
			nn.Linear(hidden[1], num_actions),
			nn.Tanh()
		)
	def forward(self, x):
		span = self.env_action.high[0]-self.env_action.low[0]
		return span*self.net(x) + self.env_action.low[0]

class Critic(nn.Module):
	def __init__(self, num_states, num_actions, hidden=[400, 300]):
		super(Critic, self).__init__()
		self.state_net = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU()
		)
		self.state_action_net = nn.Sequential(
			nn.Linear(hidden[0] + num_actions, hidden[1]),
			nn.ReLU(),
			nn.Linear(hidden[1], 1)
		)
	def forward(self, state, action):
		state_prime = self.state_net(state)
		state_action = torch.cat([state_prime, action], dim=-1)
		return self.state_action_net(state_action).squeeze()

def soft_update(src, dst, tau):
	with torch.no_grad():
		for src_param, dst_param in zip(src.parameters(), dst.parameters()):
			dst_param.copy_(tau * src_param + (1.0 - tau) * dst_param)

def copy_weights(src, dst):
	with torch.no_grad():
		for src_param, dst_param in zip(src.parameters(), dst.parameters()):
			dst_param.copy_(src_param.clone())

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.uniform_(m.bias.data)



if __name__ == '__main__':
	batch_size = 64
	gamma = 0.99
	tau = 1e-3
	num_warmup = 5000
	num_train = 60000
	num_eval = 5000
	buffer_length = 30000

	env = gym.make('Pendulum-v0')
	# env = gym.make('MountainCarContinuous-v0')
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]
		
	policy = Actor(num_states, num_actions, env.action_space).train()
	policy.apply(weights_init)
	policy_target = Actor(num_states, num_actions, env.action_space)
	copy_weights(policy, policy_target)
	
	critic = Critic(num_states, num_actions).train()
	critic.apply(weights_init)
	critic_target = Critic(num_states, num_actions)
	copy_weights(critic, critic_target)

	critic_loss = nn.MSELoss()

	noise = Noise(env.action_space)
	noise.reset()
	buffer = ReplayBuffer(buffer_length)
	state = env.reset()
	state = torch.from_numpy(state).to(dtype=torch.float32)

	opt_critic = torch.optim.Adam(critic.parameters(),lr=1e-3)
	opt_policy = torch.optim.Adam(policy.parameters(),lr=1e-4)
	policy.train()
	critic.train()
	# Figure and figure data setting
	plot_loss = []
	plot_Q = []
	score = []
	
	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
	
	episode = 0
	for training_step in range(num_train + buffer.max_length + num_eval):
		
		action = policy(state).detach()
		action = noise.add_noise(action, training_step)
		state_next, reward, term, _ = env.step(action.cpu().numpy())
		state_next = torch.from_numpy(state_next).to(dtype=state.dtype, device=state.device)
		buffer.append(state, action, reward, state_next, term)


		if training_step>num_warmup and training_step<=(num_train+num_warmup):
			b_state, b_action, b_reward, b_state_next, b_term = buffer.sample(batch_size)
			action_target = policy_target(b_state_next)
			Q_prime = critic_target(b_state_next, action_target).detach()

			opt_critic.zero_grad()
			Q = critic(b_state, b_action)
			L_critic = critic_loss(Q, b_reward + gamma*Q_prime*(1.0-b_term))
			L_critic.backward()
			opt_critic.step()
			
			opt_policy.zero_grad()
			action = policy(b_state)
			L_Q = -1.0*critic(b_state, action).mean()
			L_Q.backward()
			opt_policy.step()

			soft_update(critic, critic_target, tau)
			soft_update(policy, policy_target, tau)

			plot_loss.append(L_critic.item())
			plot_Q.append(L_Q.item())
		
		elif training_step>(num_train+num_warmup):
			env.render()

		state = state_next
		score.append(reward)
		
		if term:
			state = env.reset()
			state = torch.from_numpy(state).to(dtype=torch.float32)
			print(f'Step: {training_step} / Episode: {episode} / Score: {np.average(score)}')
			
			ax1.plot([episode], [np.average(score)], '*')
			ax1.set_ylabel('Score')
			
			if len(plot_loss)>0:
				ax2.plot([episode], [np.average(plot_loss)], 'o')
				ax2.set_ylabel('Loss')

				ax3.plot([episode], np.average(plot_Q), 'd')
				ax3.set_ylabel('Q-value')

			plt.draw()
			plt.pause(0.000001)

			plot_loss = []
			plot_Q = []
			score = []
			episode += 1

	env.close()