import numpy as np
from collections import defaultdict
import random
import pickle
from env import EphemeralTicTacToeEnv
from visualization import Visualizer
import pygame

# Configuration switches
GUI = True  # Set to True to enable GUI during training
EPISODES = 50000  # Number of training episodes
VISUALIZE_EVERY = 1  # Visualize every N episodes if GUI is True

def hash_state(obs):
    """Convert observation to a hashable state."""
    board_channel = obs[:, :, 0]
    ages_channel = obs[:, :, 1]
    return (''.join("X" if x == 1 else "O" if x == -1 else " " for x in board_channel.flatten()),
            ''.join(map(str, ages_channel.flatten().astype(int))))

def train_against_random(episodes=EPISODES, gamma=0.95, epsilon=1.0, decay_rate=0.9995, gui=GUI, visualize_every=VISUALIZE_EVERY):
    """Train a Q-learning agent against random moves."""
    env = EphemeralTicTacToeEnv()
    visualizer = Visualizer(gui=gui)
    q_table = defaultdict(lambda: np.zeros(9))
    total_wins = {"Agent": 0, "Random": 0, "Draw": 0}
    max_steps = 40
    game_history = []

    for episode in range(episodes):
        current_epsilon = epsilon * (decay_rate ** episode)
        if episode % 100 == 0 or episode == 0:
            avg_reward = sum([r for _, _, r, _, _, _ in game_history]) / max(1, len(game_history)) if game_history else 0
            print(f"Episode: {episode}/{episodes}, Avg Reward: {avg_reward:.3f}, Epsilon: {current_epsilon:.3f}")

        obs = env.reset(starting_player="X")  # Agent as X
        done = False
        step_count = 0
        visualize = gui and (episode % visualize_every == 0)
        game_history = []

        while not done and step_count < max_steps:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            if env.current_player == "X":  # Agent's turn
                state_key = hash_state(obs)
                if random.random() < current_epsilon:
                    action = random.choice(legal_actions)
                else:
                    q_values = q_table.get(state_key, np.zeros(9))
                    action = max(legal_actions, key=lambda x: q_values[x])
            else:  # Random player's turn
                action = random.choice(legal_actions)

            next_obs, reward, done, info = env.step(action)
            game_history.append((obs.copy(), action, reward, next_obs.copy(), done, env.current_player))
            if visualize:
                visualizer.refresh(next_obs, reward, done, info, env, [None, None, None])
                pygame.time.wait(500)
            obs = next_obs
            step_count += 1

        if done and reward == 1:
            winner = info["player"]
            total_wins["Agent" if winner == "X" else "Random"] += 1
        elif done:
            total_wins["Draw"] += 1

        update_q_table(game_history, q_table, alpha=0.5, gamma=gamma)

        if gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            if visualize:
                pygame.display.flip()
                pygame.time.wait(1000)

    print(f"\nTraining Complete!")
    print(f"Total Wins - Agent: {total_wins['Agent']}, Random: {total_wins['Random']}, Draw: {total_wins['Draw']}")

    with open("q_table_agent.pkl", "wb") as f:
        pickle.dump(dict(q_table), f)

    if gui:
        pygame.quit()

    return q_table

def update_q_table(game_history, q_table, alpha=0.5, gamma=0.95):
    """Update Q-table with adjusted rewards and decaying learning rate."""
    for i, (state, action, reward, next_state, done, player) in enumerate(game_history):
        if player != "X":  # Only update for agent's moves
            continue
        state_key = hash_state(state)
        next_state_key = hash_state(next_state)
        current_alpha = alpha / (1 + 0.01 * i)

        if state_key not in q_table:
            q_table[state_key] = np.zeros(9)
        if next_state_key not in q_table:
            q_table[next_state_key] = np.zeros(9)

        adjusted_reward = reward
        if done and reward == 1 and player != game_history[-1][5]:
            adjusted_reward = -1

        current_q = q_table[state_key][action]
        best_next_q = np.max(q_table[next_state_key]) if not done else 0
        new_q = (1 - current_alpha) * current_q + current_alpha * (adjusted_reward + gamma * best_next_q)
        q_table[state_key][action] = new_q

    with open("q_table_agent.pkl", "wb") as f:
        pickle.dump(dict(q_table), f)

if __name__ == "__main__":
    q_table = train_against_random()