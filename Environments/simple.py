import os
import sys

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import gym
from gym.spaces import Discrete, Box

import numpy as np
from InvPendulum import InvPendulumEnv

import argparse

def get_mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i<len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.shape[0]
    
    # logits_net = get_mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    # def get_policy(obs):
    #     logits = logits_net(obs)
    #     return Categorical(logits=logits)
    # def get_action(obs):
    #     return get_policy(obs).sample().item()
    # def compute_loss(obs, act, weights):
    #     logp = get_policy(obs).log_prob(act)
    #     return -(logp * weights).mean()

    #continuos control
    mu_net = get_mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    sigma_net = get_mlp(sizes=[obs_dim]+hidden_sizes+[n_acts], output_activation=nn.Sigmoid)
    def get_policy(obs):
        log_mu = mu_net(obs)
        log_sigma = sigma_net(obs) + 1E-5
        return Normal(loc=log_mu, scale=log_sigma)
    def get_action(obs):
        return np.array([torch.clamp(get_policy(obs).sample(), -2.0, 2.0)])
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights.unsqueeze(dim=1)).mean()

    optimizer = Adam(list(mu_net.parameters()) + list(sigma_net.parameters()), lr=lr)
    def train_one_epoch():
        batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = [], [], [], [], []

        obs = env.reset()
        done = False
        ep_rews = []

        while True:
            if render: env.render()
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_weights += list(reward_to_go(ep_rews))
                obs, ep_rews, done = env.reset(), [], False
                if len(batch_obs) > batch_size:
                    break
        
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                    act=torch.as_tensor(batch_acts, dtype=torch.float32),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--env_name', '--env', type=str, default='Pendulum-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)