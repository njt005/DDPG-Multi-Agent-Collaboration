import os
import typing as t
import numpy as np
import pandas as pd
import torch
from collections import deque
from unityagents import UnityEnvironment

from agent import Agent
from plotting import create_reward_plot, create_seaborn_barplot

# Agent parameters
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-2
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
UPDATE_EVERY = 20
N_UPDATES = 15
HIDDEN_SIZES_ACTOR = (256, 128, 64)
HIDDEN_SIZES_CRITIC_S = (256,)
HIDDEN_SIZES_CRITIC_SA = (128, 64)
WEIGHT_DECAY_CRITIC = 0

# Environment - update based on Unity environment location
# Put folder in same directory as this file
ENV_PATH = "Tennis_Windows_x86_64/Tennis.exe"

# Training parameters
N_EPISODES = 2_000
REWARD_GOAL = 0.5
MAX_ITER_EPISODE = 1_000
EPSILON = 1.0
EPSILON_DECAY = 1e-3
EPSILON_MIN = 1e-2
N_LAGS = None
PLOT_RESULTS = True
N_TRAINING_SEEDS = 4
ADD_NOISE = True


def train_agent(
    agent: Agent,
    env: UnityEnvironment,
    brain_name: str,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    n_episodes: int,
    max_iter_episode: int,
    reward_goal: int,
    n_lags: t.Optional[int],
    plot_reward: bool = True,
    save_path: t.Optional[str] = None,
):
    """train_agent

    Takes a DQN agent and environment as input and trains the agent.

    Parameters
    ----------
    agent : Agent
        agent
    env : UnityEnvironment
        Unity environment
    brain_name : str
        brain name
    epsilon : float
        exploration / exploitation tradeoff parameter
    epsilon_decay : float
        rate to decay epsilon
    epsilon_min : float
        min espsilon
    n_episodes : int
        number of episodes to train
    max_iter_episode : int
        maximum number of iterations in an episode
    reward_goal : int
        reward goal to stop training
    n_lags : t.Optional[int]
        number of lags to lag current state
        if None - no lags
    plot_reward : bool
        plot reward once finished training
    save_path : t.Optional[str]
        path to save training results
    """
    scores = []
    scores_window = deque(maxlen=100)
    for episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        states = []
        state = env_info.vector_observations  # state for each agent
        n_agents = len(env_info.agents)
        score = np.zeros(n_agents)

        # Update epsilon based on decay rate
        if epsilon_decay is not None:
            epsilon = np.max([epsilon - epsilon * epsilon_decay, epsilon_min])

        for _ in range(max_iter_episode):
            # Store current state and add lags
            states.append(state)
            if n_lags is not None:
                if len(states) <= n_lags:
                    for _ in range(n_lags + 1):
                        states.append(state)

                state = np.hstack(states[-n_lags - 1 :])

            # Take action and update networks
            action = agent.act(state=state, add_noise=ADD_NOISE, epsilon=epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            if n_lags is not None:
                next_state_save = np.hstack(states[-n_lags:] + [next_state])
            else:
                next_state_save = next_state
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state_save, done)
            score += reward
            state = next_state

            if np.any(done):
                break

        # Append scores
        scores_window.append(np.max(score))  # Get max out of the agents
        scores.append(score)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}, Epsilon: {:.4f}".format(
                episode, np.mean(scores_window), epsilon
            ),
            end="",
        )

        if episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= reward_goal:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_window)
                )
            )
            if save_path is not None:
                model_save_path = os.path.join(save_path, "models", f"{agent.seed}")
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                torch.save(
                    agent.actor_local.state_dict(),
                    os.path.join(model_save_path, f"actor_n{episode}.pt"),
                )

                torch.save(
                    agent.critic_local.state_dict(),
                    os.path.join(model_save_path, f"critic_n{episode}.pt"),
                )
            break

    if plot_reward:
        if save_path is not None:
            results_save_path = os.path.join(save_path, "results", f"{agent.seed}")
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
        create_reward_plot(
            scores=np.array(scores).T,
            save_path=os.path.join(results_save_path, f"scores_n{episode}.png"),
        )

    # Log training run
    log_dict = {
        "Seed": agent.seed,
        "Reward Goal": reward_goal,
        "N Lags": n_lags,
        "N Episodes": episode,
    }

    return pd.DataFrame([log_dict])


def main():
    # Set up paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_full_path = os.path.join(base_path, ENV_PATH)

    # Results to save df and figures
    results_save_path = os.path.join(base_path, "results")
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # Initialize environment and get env info to initialize Agent
    env = UnityEnvironment(file_name=env_full_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    if N_LAGS is not None:
        state_size = state_size * (N_LAGS + 1)
    action_size = brain.vector_action_space_size

    df = pd.DataFrame()
    for i in range(N_TRAINING_SEEDS):
        print(
            f"--------------------Training seed {i+1}/{N_TRAINING_SEEDS}--------------------"
        )
        # Initialize agent
        agent = Agent(
            n_agents=len(env_info.agents),
            state_size=state_size,
            action_size=action_size,
            hidden_sizes_actor=HIDDEN_SIZES_ACTOR,
            hidden_sizes_critic_s=HIDDEN_SIZES_CRITIC_S,
            hidden_sizes_critic_sa=HIDDEN_SIZES_CRITIC_SA,
            learning_rate_actor=LEARNING_RATE_ACTOR,
            learning_rate_critic=LEARNING_RATE_CRITIC,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            update_every=UPDATE_EVERY,
            n_updates=N_UPDATES,
            tau=TAU,
            weight_decay_critic=WEIGHT_DECAY_CRITIC,
            seed=i + 1,
        )

        df_seed = train_agent(
            agent=agent,
            env=env,
            brain_name=brain_name,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            n_episodes=N_EPISODES,
            max_iter_episode=MAX_ITER_EPISODE,
            reward_goal=REWARD_GOAL,
            n_lags=N_LAGS,
            plot_reward=PLOT_RESULTS,
            save_path=base_path,
        )
        df = pd.concat([df, df_seed], axis=0, ignore_index=True)
        df.to_csv(os.path.join(results_save_path, "training_log.csv"))

    if PLOT_RESULTS:
        create_seaborn_barplot(
            df=df,
            x="Seed",
            y="N Episodes",
            save_path=os.path.join(results_save_path, "average_training_episodes.png"),
        )

    env.close()


if __name__ == "__main__":
    main()
