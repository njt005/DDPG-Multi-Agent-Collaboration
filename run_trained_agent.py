import os
import numpy as np
import torch
from unityagents import UnityEnvironment

from model import Actor
from train_agent import N_LAGS, HIDDEN_SIZES_ACTOR, ENV_PATH

# Which trained agent to load
SEED = 4


def main():
    # Initialize environment and get env info to initialize Agent
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_full_path = os.path.join(base_path, ENV_PATH)
    env = UnityEnvironment(file_name=env_full_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    if N_LAGS is not None:
        state_size = state_size * (N_LAGS + 1)
    action_size = brain.vector_action_space_size

    # Get trained agent
    policy = Actor(
        state_size=state_size, action_size=action_size, hidden_sizes=HIDDEN_SIZES_ACTOR
    )
    model_path = os.path.join(base_path, "models", str(SEED))
    model_name = os.listdir(model_path)[0]  # Just get first model if more than 1
    policy.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    policy.eval()

    # Watch single episode
    state = env_info.vector_observations
    score = np.zeros(len(env_info.agents))
    states = []
    while True:
        states.append(state)
        if N_LAGS is not None:
            if len(states) <= N_LAGS:
                for _ in range(N_LAGS + 1):
                    states.append(state)
            state = np.hstack(states[-N_LAGS - 1 :])
        action = policy(torch.FloatTensor(state)).detach().cpu().numpy()
        action = np.clip(action, -1, 1)  # clip just in case
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        score += reward
        state = next_state
        if np.any(done):
            break

    print(f"Single episode score: {np.mean(score):.2f}")
    env.close()


if __name__ == "__main__":
    main()
