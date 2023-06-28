import typing as t
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


class Agent:
    """
    Agent that utilizes a linear MLP Q-Network to approximate
    a continuous policy and and state-action value in environment
    """

    def __init__(
        self,
        n_agents: int,
        state_size: int,
        action_size: int,
        hidden_sizes_actor: t.Tuple[int, ...],
        hidden_sizes_critic_s: t.Tuple[int, ...],
        hidden_sizes_critic_sa: t.Tuple[int, ...],
        learning_rate_actor: float = 1e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        update_every: int = 20,
        n_updates: int = 10,
        tau: float = 1e-3,
        weight_decay_critic: float=0.0001,
        seed: int = 5,
    ):
        """__init__

        Initialize Agent with DDPG structure.

        Parameters
        ----------
        n_agents : int
            number of agents to form super agent
        state_size : int
            size of input state
        action_size : int
            size of output (number of available actions)
        hidden_size_actor : t.Tuple[int, ...]
            hidden layer sizes for actor
        hidden_size_critic_s : t.Tuple[int, ...]
            hidden layer sizes for critic state pre-processing
        hidden_size_critic_sa : t.Tuple[int, ...]
            hidden layer sizes for critic state and action
        learning_rate_actor : float
            learning rate for updating actor weights
        learning_rate_critic : float
            learning rate for updating critic weights
        gamma : float
            reward discount
        batch_size : int
            size of batches used for training
        update_every : int
            update target network this often
        n_updates : int
            make n_update random samples to update target network
        tau : float
            soft update parameter
        weight_decay_critic : float
            decay critic weights
        seed : int, optional
            random seed for network, by default 5
        """
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes_actor = hidden_sizes_actor
        self.hidden_sizes_critic_s = hidden_sizes_critic_s
        self.hidden_sizes_critic_sa = hidden_sizes_critic_sa
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.n_updates = n_updates
        self.tau = tau
        self.seed = seed

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actors
        self.actor_local = Actor(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=self.hidden_sizes_actor,
            seed=self.seed,
        ).to(self.device)

        self.actor_target = Actor(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=self.hidden_sizes_actor,
            seed=self.seed,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=learning_rate_actor
        )

        # Critics
        self.critic_local = Critic(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes_s=self.hidden_sizes_critic_s,
            hidden_sizes_sa=self.hidden_sizes_critic_sa,
            seed=self.seed,
        ).to(self.device)

        self.critic_target = Critic(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes_s=self.hidden_sizes_critic_s,
            hidden_sizes_sa=self.hidden_sizes_critic_sa,
            seed=self.seed,
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=learning_rate_critic, weight_decay=weight_decay_critic
        )

        # Replay memory
        self.memory = ReplayBuffer(
            action_size=action_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
        )

        # Noise process
        self.noise = OUNoise((n_agents, action_size), seed)

        # Initialize time step
        self.t_step = 0

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """step

        Add state, action, reward, next_state, and done to memory
        Update network weights if step is an update_every step

        Parameters
        ----------
        state : np.ndarray
            state of each agent
        action : int
            action each agent took
        reward : int
            reward received from each agent action
        next_state : np.ndarray
            next state of each agent
        done : bool
            end of episode for each agent
        """
        # Save experience in replay memory
        for i_agent in range(self.n_agents):
            self.memory.add(state[i_agent], action[i_agent], reward[i_agent], next_state[i_agent], done[i_agent])

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.n_updates):
                    experiences = self.memory.sample(batch_size=self.batch_size)
                    self.learn(experiences=experiences, gamma=self.gamma, tau=self.tau)
            

    def act(self, state: np.ndarray, add_noise: bool = True, epsilon : float = 0) -> int:
        """act

        Get action from current policy with added noise.

        Parameters
        ----------
        state : np.ndarray
            current state of each agent
        add_noise : bool
            add noise to network and action
        epsilon : float
            scaling of noise added

        Returns
        -------
        int
            action for agent to take
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            # if add_noise:
            #     # Add Gaussian noise to network weights
            #     for param in self.actor_local.parameters():
            #         param.add_(torch.randn(param.size()) * epsilon*0.01)
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (epsilon*self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(
        self, experiences: t.Tuple[torch.Tensor], gamma: float, tau: float
    ) -> None:
        """learn

        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences : t.Tuple[torch.Tensor]
            tuple of (s, a, r, s', done) tuples
        gamma : float
           reward discount factor
        tau : float
            soft update parameter
        """
        states, actions, rewards, next_states, dones = experiences

        #### ------ Update Critic ------ ####
        actions_ = self.actor_target(next_states)
        Q_targets_ = self.critic_target(next_states, actions_)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_ * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.critic_local(states, actions)

        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        #### ------ Update Actor ------ ####
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #### ------ Update Target Networks ------ ####
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

    def soft_update(
        self,
        local_model: t.Any,
        target_model: t.Any,
        tau: float,
    ):
        """soft_update

        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        Parameters
        ----------
        local_model : Network
            weights copied from local model to target
        target_model : Network
            target model that receives copies of weights from local network
        tau : float
            soft update parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, action_size: int, buffer_size: int, batch_size: int, seed: int = 5
    ):
        """__init__

        Initialize a replay buffer object

        Parameters
        ----------
        action_size : int
            size of action space
        buffer_size : int
            size of replay buffer to sample from
        batch_size : int
            size of batches
        seed : int, optional
            random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """add

        Add experience to memory

        Parameters
        ----------
        state : np.ndarray
            state of agent
        action : int
            action agent took
        reward : int
            reward received from action
        next_state : np.ndarray
            next state of agent
        done : bool
            end of episode
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size: int) -> t.Tuple[torch.Tensor]:
        """sample

        Sample from memory to the length of the batch_size

        Parameters
        ----------
        batch_size : int
            size of batches

        Returns
        -------
        t.Tuple[float.Tensor]
            tuple of sampled (s, a, r, s', done)
        """
        sampled_idx = np.random.choice(np.arange(len(self.memory)), batch_size)
        experiences = [self.memory[idx] for idx in sampled_idx]
        experiences = [
            experience for experience in experiences if experience is not None
        ]

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2) -> None:
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
