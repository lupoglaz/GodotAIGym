import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Environments.InvPendulum import InvPendulumEnv
import numpy as np
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()
		self.data = []
		
		self.fc1 = nn.Linear(4,256)
		self.fc_pi = nn.Linear(256,4)
		self.fc_v = nn.Linear(256,2)
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		
	def pi(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc_pi(x)
		x = x.view(-1, 2, 2).squeeze()
		
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

		if s.ndimension() == 2:
			return

		pi = self.pi(s)
		pi_a = pi.gather(2, a.unsqueeze(dim=2)).squeeze()
		loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

		self.optimizer.zero_grad()
		loss.mean().backward()
		self.optimizer.step()         
	  
def main():  
	env = InvPendulumEnv(exec_path="Environments/InvPendulum.x86_64", env_path="Environments/InvPendulum.pck")
	model = ActorCritic()    
	n_rollout = 10
	print_interval = 40
	score = 0.0

	for n_epi in range(10000):
		done = False
		s = env.reset()
		while not done:
			for t in range(n_rollout):
				prob = model.pi(s)
				m = Categorical(prob)
				a = m.sample().to(dtype=torch.int, device='cpu')
				s_prime, r, done, info = env.step(a)
				model.put_data((s,a,r,s_prime,done))
				
				s = s_prime
				score += r
				
				if done:
					break                     
			
			model.train_net()
			
		if n_epi%print_interval==0 and n_epi!=0:
			print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
			score = 0.0
	env.close()

if __name__ == '__main__':
	main()