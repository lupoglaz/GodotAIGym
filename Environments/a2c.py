import torch
from torch import nn
from collections import namedtuple, deque, OrderedDict
from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch import optim
from torch.optim import Optimizer
import gym

import numpy as np
import argparse

import pytorch_lightning as pl

class DQN(nn.Module):
	def __init__(self, obs_size: int, n_actions: int, hidden_size: int=128):
		super(DQN, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
	def forward(self, x):
		return self.net(x)

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
	def __init__(self, capacity:int) -> None:
		self.buffer = deque(maxlen=capacity)
	def __len__(self):
		return len(self.buffer)
	def append(self, experience:Experience) -> None:
		self.buffer.append(experience)
	def sample(self, batch_size:int) -> Tuple:
		indices = np.random.choice(len(self), batch_size, replace=False)
		states, actions, rewards, dones, new_states = zip(*[self.buffer[idx] for idx in indices])
		return (np.array(states, dtype=np.float32), np.array(actions), 
				np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), 
				np.array(new_states, dtype=np.float32))

class RLDataset(IterableDataset):
	def __init__(self, buffer:ReplayBuffer, sample_size: int = 200) -> None:
		self.buffer = buffer
		self.sample_size = sample_size
	def __iter__(self) -> Tuple:
		states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(dones)):
			yield states[i], actions[i], rewards[i], dones[i], new_states[i]
	
class Agent:
	def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
		self.env = env
		self.replay_buffer = replay_buffer
		self.reset()
	
	def reset(self):
		self.state = self.env.reset()

	def get_action(self, net:nn.Module, epsilon:float, device:str) -> int:
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state = torch.tensor([self.state], dtype=torch.float32)
			if not device in ['cpu']:
				state = state.cuda(device)
			
			q_values = net(state)
			_, action = torch.max(q_values, dim=1)
			action = action.item()
		return action
	
	@torch.no_grad()
	def play_step(self, net:nn.Module, epsilon: float=0.0, device: str='cpu') -> Tuple[float, bool]:
		action = self.get_action(net, epsilon, device)
		new_state, reward, done, _ = self.env.step(action)
		exp = Experience(self.state, action, reward, done, new_state)
		self.replay_buffer.append(exp)
		self.state = new_state
		if done:
			self.reset()
		return reward, done

class DQNLightning(pl.LightningModule):
	def __init__(self, hparams: argparse.Namespace) -> None:
		super().__init__()
		self.hparams = hparams

		self.env = gym.make(self.hparams.env)
		obs_size = self.env.observation_space.shape[0]
		n_actions = self.env.action_space.n

		self.net = DQN(obs_size, n_actions)
		self.target_net = DQN(obs_size, n_actions)

		self.buffer = ReplayBuffer(self.hparams.replay_size)
		self.agent = Agent(self.env, self.buffer)
		self.total_reward = 0
		self.episode_reward = 0
		self.populate(self.hparams.warm_start_steps)

	def populate(self, steps: int=1000):
		for i in range(steps):
			self.agent.play_step(self.net, epsilon=1.0)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		output = self.net(x)
		return output

	def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
		states, actions, rewards, dones, new_states = batch
		state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

		with torch.no_grad():
			next_state_values = self.target_net(new_states).max(1)[0]
			next_state_values[dones] = 0
			# next_state_values = next_state_values.detach()

		expected_state_action_values = next_state_values * self.hparams.gamma + rewards
		return nn.MSELoss()(state_action_values, expected_state_action_values)

	def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
		device = self.get_device(batch)
		epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1/self.hparams.eps_last_frame)

		reward, done = self.agent.play_step(self.net, epsilon, device)
		self.episode_reward += reward

		loss = self.dqn_mse_loss(batch)

		if self.trainer.use_dp or self.trainer.use_ddp2:
			loss = loss.unsqueeze(0)
		
		if done:
			self.total_reward = self.episode_reward
			self.episode_reward = 0
		
		if self.global_step % self.hparams.sync_rate == 0:
			self.target_net.load_state_dict(self.net.state_dict())

		log = {'total_reward': torch.tensor(self.total_reward).to(device),
				'reward': torch.tensor(reward).to(device),
				'train_loss': loss}
		status = {'steps': torch.tensor(self.global_step).to(device),
				'total_reward': torch.tensor(self.total_reward).to(device)}
		return OrderedDict({'loss': loss, 'log':log, 'progress_bar': status})

	def configure_optimizers(self) -> List[Optimizer]:
		optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
		return optimizer

	def __dataloader(self) -> DataLoader:
		dataset = RLDataset(self.buffer, self.hparams.episode_length)
		dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
		return dataloader
	
	def train_dataloader(self) -> DataLoader:
		return self.__dataloader()
	
	def get_device(self, batch) -> str:
		return batch[0].device.index if self.on_gpu else 'cpu'

def main(hparams) -> None:
	model = DQNLightning(hparams)
	trainer = pl.Trainer(gpus=1, max_epochs=500, early_stop_callback=None, val_check_interval=100)
	trainer.fit(model)

if __name__=='__main__':
	torch.manual_seed(0)
	np.random.seed(0)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
	parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
	parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--sync_rate", type=int, default=10,
						help="how many frames do we update the target network")
	parser.add_argument("--replay_size", type=int, default=1000,
						help="capacity of the replay buffer")
	parser.add_argument("--warm_start_size", type=int, default=1000,
						help="how many samples do we use to fill our buffer at the start of training")
	parser.add_argument("--eps_last_frame", type=int, default=1000,
						help="what frame should epsilon stop decaying")
	parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
	parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
	parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
	parser.add_argument("--max_episode_reward", type=int, default=200,
						help="max episode reward in the environment")
	parser.add_argument("--warm_start_steps", type=int, default=1000,
						help="max episode reward in the environment")

	args, _ = parser.parse_known_args()

	main(args)