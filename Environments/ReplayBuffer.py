import random
import torch
import numpy as np


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
	
	def get_histograms(self):
		state, action, reward, state_next, term = self.sample(1000)
		ah = np.histogram(action.cpu().numpy(), bins=60)
		rh = np.histogram(reward.cpu().numpy(), bins=60)
		trh = np.histogram(np.linalg.norm((state_next - state).cpu().numpy(), axis=1))
		return ah, rh, trh

	def __len__(self):
		return len(self.buffer)