import pickle
import pygame
import numpy as np
from env import EphemeralTicTacToeEnv
from visualization import Visualizer
from train import hash_state, update_q_table
import random

# Configuration switches
GUI = True  # Set to False to disable GUI
NRUNS = 1  # Number of games to simulate

# Visualization delays (in milliseconds)
MOVE_DELAY = 500
END_GAME_DELAY = 1000

def load_q_table():
    """Load trained Q-table."""
    try:
        with open("q_table_agent.pkl", "rb") as f:
            q_table = pickle.load(f)
        return q_table
    except FileNotFoundError:
        print("Error: Q-table not found. Please run train.py first.")
        raise SystemExit

class QTablePlayer:
    """Player that uses Q-table for action selection."""
    def select_action(self, env, obs, q_table):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
        state_key = hash_state(obs)
        q_values = q_table.get(state_key, np.zeros(9))
        return max(legal_actions, key=lambda x: q_values[x])

def play_human_vs_ai(gui=GUI, game_number=1):
    """Play a game between human (X) and AI (O), updating Q-table after game."""
    pygame.init()
    q_table = load_q_table()
    env = EphemeralTicTacToeEnv()
    visualizer = Visualizer(gui=gui)
    obs = env.reset(starting_player="X")  # Human as X, AI as O
    ai_player = QTablePlayer()
    action_history = [None, None, None]
    game_history = []
    done = False
    step_count = 0
    max_steps = 20

    # Render initial board state
    if gui:
        visualizer.refresh(obs, 0, False, {}, env, action_history)

    print(f"Game {game_number}: Human as X vs AI as O")

    while not done and step_count < max_steps:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break

        if env.current_player == "X":  # Human's turn
            action = None
            if gui:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        visualizer.close()
                        raise SystemExit
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        row, col = y // 200, x // 200
                        if 0 <= row < 3 and 0 <= col < 3:
                            action = row * 3 + col
                            if action not in legal_actions:
                                action = None
                if action is None:
                    visualizer.refresh(obs, 0, False, {}, env, action_history)  # Keep screen updated
                    continue
            else:
                print("Legal moves:", [(a // 3, a % 3) for a in legal_actions])
                try:
                    row, col = map(int, input("Enter row, col (e.g., 0 1): ").split())
                    action = row * 3 + col
                    if action not in legal_actions:
                        print("Illegal move!")
                        continue
                except (ValueError, IndexError):
                    print("Invalid input!")
                    continue
        else:  # AI's turn
            action = ai_player.select_action(env, obs, q_table)

        if action is None:
            if env.current_player == "X":
                continue
            print("No valid action, ending game.")
            break

        next_obs, reward, done, info = env.step(action)
        game_history.append((obs.copy(), action, reward, next_obs.copy(), done, env.current_player))
        if gui:
            visualizer.refresh(next_obs, reward, done, info, env, action_history)
            pygame.time.wait(MOVE_DELAY)
            visualizer.check_quit()
        obs = next_obs
        step_count += 1

    if done:
        if reward == 1:
            winner = info['player']
            winner_name = "Human" if winner == "X" else "AI"
            print(f"{winner_name} ({winner}) wins!")
        else:
            print("Draw!")
    else:
        print("Game ended with no winner (max steps reached).")

    # Update Q-table with game history
    update_q_table(game_history, q_table, alpha=0.1, gamma=0.95)

    if gui:
        pygame.display.flip()
        pygame.time.wait(END_GAME_DELAY)
        visualizer.check_quit()
    visualizer.close()
    return None if not done else winner

def simulate_games(nruns=NRUNS, gui=GUI):
    """Simulate multiple human vs AI games."""
    wins = [0, 0, 0]  # [draws, human wins, AI wins]

    for n in range(1, nruns + 1):
        print(f"\nRun {n}")
        result = play_human_vs_ai(gui=gui, game_number=n)
        if result is None:
            wins[0] += 1
            print("Draw!")
        elif result == "X":
            wins[1] += 1
            print("Human wins!")
        else:
            wins[2] += 1
            print("AI wins!")
        
        print(f"Stats: Draws={wins[0]}, Human={wins[1]}, AI={wins[2]}")
        print(f"Win rates: {', '.join([f'{v/n:.2f}' for v, n in zip(wins, [n, n, n])])}")

    return wins

if __name__ == "__main__":
    simulate_games()