import torch
import numpy as np


class AnnealedGaussianProcess(object):
	def __init__(self, mu, sigma, sigma_min, n_annealing):
		self.mu = mu 
		self.sigma = sigma 
		self.n_steps = 0

		if not sigma_min is None:
			self.m = -float(sigma - sigma_min) / float(n_annealing)
			self.c = sigma
			self.sigma_min = sigma_min
		else:
			self.m = 0
			self.c = sigma
			self.sigma_min = sigma 
	
	def reset_states(self):
		pass
	
	@property
	def current_sigma(self):
		return max(self.sigma_min, self.m * float(self.n_steps) + self.c)

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
	def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_annealing=1000):
		super(OrnsteinUhlenbeckProcess, self).__init__(mu, sigma, sigma_min, n_annealing)
		self.theta = theta
		self.mu = mu
		self.dt = dt
		self.x0 = x0
		self.size = size
		self.n_steps = 0

	def reset_states(self):
		if self.x0 is None:
			self.x_prev = np.zeros(self.size)
		else:
			self.x_prev = self.x0
	
	def state_update(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)
	
	def sample(self):
		x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.current_sigma*np.sqrt(self.dt)*np.random.normal(size=self.size)
		self.x_prev = x
		self.n_steps += 1
		return x
	