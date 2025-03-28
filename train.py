# train.py
import numpy as np
from collections import defaultdict
import random
import pickle
from env import EphemeralTicTacToeEnv
from visualization import Visualizer

def hash_state(obs):
    """Convert observation to a hashable state."""
    board_channel = obs[:, :, 0]
    ages_channel = obs[:, :, 1]
    return (''.join("X" if x == 1 else "O" if x == -1 else " " for x in board_channel.flatten()),
            ''.join(map(str, ages_channel.flatten().astype(int))))

def train_agents(episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999, gui=False, visualize_every=100):
    """Train Q-learning agents for X and O."""
    env = EphemeralTicTacToeEnv()
    visualizer = Visualizer(gui=gui)
    q_table_x = defaultdict(lambda: np.zeros(9))
    q_table_o = defaultdict(lambda: np.zeros(9))
    total_rewards = {"X": 0.0, "O": 0.0}
    episode_rewards = {"X": [], "O": []}
    max_steps = 20

    for episode in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        current_epsilon = epsilon * (decay_rate ** episode)
        visualize = gui and (episode % visualize_every == 0)
        episode_reward = {"X": 0.0, "O": 0.0}
        action_history = [None, None, None]

        if visualize:
            print(f"Visualizing Episode {episode}/{episodes} (Starting Player: {env.game.current_player})")
        elif episode % 1000 == 0:
            print(f"Training Episode {episode}/{episodes}")

        while not done and step_count < max_steps:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            state_key = hash_state(obs)
            q_table = q_table_x if env.game.current_player == "X" else q_table_o

            action = (random.choice(legal_actions) if random.random() < current_epsilon else
                      max(legal_actions, key=lambda x: q_table[state_key][x]))

            next_obs, reward, done, info = env.step(action)
            episode_reward[env.game.current_player] += reward

            if visualize:
                visualizer.refresh(next_obs, reward, done, info, env, action_history)

            next_state_key = hash_state(next_obs)
            q_table[next_state_key]  # Ensure key exists
            current_q = q_table[state_key][action]
            best_next_q = np.max(q_table[next_state_key])
            q_table[state_key][action] = (1 - 0.1) * current_q + 0.1 * (reward + gamma * best_next_q)

            obs = next_obs
            step_count += 1
            if visualize:
                visualizer.check_quit()

        for player in ["X", "O"]:
            total_rewards[player] += episode_reward[player]
            episode_rewards[player].append(episode_reward[player])

    print(f"\nTraining Complete!")
    for player in ["X", "O"]:
        print(f"Total Reward for {player}: {total_rewards[player]:.2f}")
        print(f"Average Reward per Episode for {player}: {np.mean(episode_rewards[player]):.4f}")

    with open("q_table_x.pkl", "wb") as f:
        pickle.dump(dict(q_table_x), f)
    with open("q_table_o.pkl", "wb") as f:
        pickle.dump(dict(q_table_o), f)

    return q_table_x, q_table_o

if __name__ == "__main__":
    q_table_x, q_table_o = train_agents(gui=False)