import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Environments.InvPendulum import InvPendulumEnv
import numpy as np
import time
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

from matplotlib import pylab as plt

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()
		self.data = []
		
		self.fc1 = nn.Linear(3,256)
		self.fc_pi = nn.Linear(256,4)
		self.fc_v = nn.Linear(256,2)
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		
	def pi(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc_pi(x)
		x = x.view(-1, 2, 2)
		
		prob = F.softmax(x, dim=(x.ndimension()-1))
		return prob
	
	def v(self, x):
		x = F.relu(self.fc1(x))
		v = self.fc_v(x)
		return v
	
	def put_data(self, transition):
		self.data.append(transition)
		
	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
		for transition in self.data:
			s,a,r,s_prime,done = transition
			s_lst.append(s)
			a_lst.append(a.to(dtype=torch.long))
			r_lst.append([r/100.0])
			s_prime_lst.append(s_prime)
			done_mask = 0.0 if done else 1.0
			done_lst.append([done_mask])
		
		s_batch = torch.stack(s_lst, dim=0)
		a_batch = torch.stack(a_lst, dim=0)
		s_prime_batch = torch.stack(s_prime_lst, dim=0)
		done_batch = torch.tensor(done_lst, dtype=torch.float)
		r_batch = torch.tensor(r_lst, dtype=torch.float)
		
		self.data = []
		return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
	def train_net(self):
		s, a, r, s_prime, done = self.make_batch()
		td_target = r + gamma * self.v(s_prime) * done
		delta = td_target - self.v(s)

		pi = self.pi(s)
		pi_a = pi.gather(2, a.unsqueeze(dim=2)).squeeze()
		loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
		L = loss.mean()

		self.optimizer.zero_grad()
		L.backward()
		self.optimizer.step()       
		return L.item()


def main():  
	env = InvPendulumEnv(exec_path="Environments/InvPendulum.x86_64", env_path="Environments/InvPendulum.pck")
	model = ActorCritic().train()
	n_rollout = 5
	print_interval = 20
	score = 0.0

	av_rewards = []
	std_rewards = []
	rewards = []
	for n_epi in range(5000):
		done = False
		s = env.reset()
		reward = 0.0
		while not done:
			for t in range(n_rollout):
				prob = model.pi(s).squeeze()
				m = Categorical(prob)
				a = m.sample().to(dtype=torch.int, device='cpu')
				s_prime, r, done, info = env.step(a)
				model.put_data((s,a,r,s_prime,done))
				
				s = s_prime
				score += r
				reward += r
				if done:
					break
			
			L = model.train_net()
		
		rewards.append(reward)

		if n_epi%print_interval==0 and n_epi!=0:
			print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
			score = 0.0
			av_rewards.append(np.average(rewards))
			std_rewards.append(np.std(rewards))
			rewards = []
	env.close()

	fig = plt.figure()
	plt.errorbar([i for i in range(len(av_rewards))], av_rewards, std_rewards)
	plt.show()



if __name__ == '__main__':
	main()