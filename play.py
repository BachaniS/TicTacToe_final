import pickle
import pygame
import numpy as np
from env import EphemeralTicTacToeEnv
from visualization import Visualizer
from train import hash_state
from config import CELL_SIZE
import random

def load_q_tables():
    """Load trained Q-tables."""
    try:
        with open("q_table_x.pkl", "rb") as f:
            q_table_x = pickle.load(f)
        with open("q_table_o.pkl", "rb") as f:
            q_table_o = pickle.load(f)
        return q_table_x, q_table_o
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train the agents first.")
        raise SystemExit

class MCTSNode:
    """Node for Monte Carlo Tree Search."""
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0

    def is_fully_expanded(self, legal_actions):
        return len(self.children) == len(legal_actions)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_reward / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for action, child in self.children.items()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        return child

def mcts_search(env, root_state, q_table, iterations=1000):
    root = MCTSNode(root_state)
    initial_legal_actions = env.get_legal_actions()

    if not initial_legal_actions:
        print("No legal actions at root state:", root_state)
        return None

    for i in range(iterations):
        node = root
        temp_env = EphemeralTicTacToeEnv()
        # Sync temp_env with root_state and original env
        temp_env.game.board = root_state[:, :, 0].copy()
        temp_env.game.ages = root_state[:, :, 1].copy()
        temp_env.game.owners = root_state[:, :, 2].copy()
        temp_env.game.current_player = env.game.current_player
        temp_env.game.move_count = env.game.move_count

        # Verify initialization
        temp_legal = temp_env.get_legal_actions()
        if set(temp_legal) != set(initial_legal_actions):
            print(f"Iteration {i}: Mismatch in legal actions - Initial: {initial_legal_actions}, Temp: {temp_legal}")
            print(f"Root state: {root_state}")
            print(f"Temp board: {temp_env.game.board}")

        # Selection
        while node.is_fully_expanded(temp_legal) and node.children:
            node = node.best_child(c_param=2.0)
            temp_env.step(node.action)
            temp_legal = temp_env.get_legal_actions()

        # Expansion
        unexplored_actions = [a for a in temp_legal if a not in node.children]
        if unexplored_actions:
            state_key = hash_state(temp_env._get_observation(temp_env.game.get_state()))
            action = max(unexplored_actions, key=lambda x: q_table.get(state_key, np.zeros(9))[x])
            next_obs, reward, done, _ = temp_env.step(action)
            node = node.expand(action, next_obs)
            if done and reward == 1:
                while node:
                    node.visits += 1
                    node.total_reward += 1
                    node = node.parent
                return action
        else:
            print(f"Iteration {i}: No unexplored actions. Legal: {temp_legal}, Children: {list(node.children.keys())}")

        # Simulation
        reward = simulate(temp_env, q_table)  # Will fix epsilon below

        # Backpropagation
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    if not root.children:
        print("Warning: No children expanded after iterations. Initial legal actions:", initial_legal_actions)
        return random.choice(initial_legal_actions)

    return max(root.children.items(), key=lambda x: x[1].visits)[0]

def simulate(env, q_table, max_steps=20, epsilon=0.3):
    step_count = 0
    done = False
    total_reward = 0

    while not done and step_count < max_steps:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        state_key = hash_state(env._get_observation(env.game.get_state()))
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            q_values = q_table.get(state_key, np.zeros(9))
            action = max(legal_actions, key=lambda x: q_values[x])
        _, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

    return total_reward

def update_q_tables(game_history, q_table_x, q_table_o, alpha=0.1, gamma=0.9):
    for state, action, reward, next_state, done, player in game_history:
        state_key = hash_state(state)
        next_state_key = hash_state(next_state)
        q_table = q_table_x if player == "X" else q_table_o

        if state_key not in q_table:
            q_table[state_key] = np.zeros(9)
        if next_state_key not in q_table:
            q_table[next_state_key] = np.zeros(9)

        adjusted_reward = reward
        if done and reward == 1 and player != game_history[-1][5]:  # Losing player
            adjusted_reward = -1
            print(f"Penalizing {player} for move {action} with reward {adjusted_reward}")

        current_q = q_table[state_key][action]
        best_next_q = np.max(q_table[next_state_key]) if not done else 0
        new_q = (1 - alpha) * current_q + alpha * (adjusted_reward + gamma * best_next_q)
        q_table[state_key][action] = new_q
        if adjusted_reward != reward:
            print(f"Q[{state_key}][{action}] updated: {current_q:.4f} -> {new_q:.4f}")

    with open("q_table_x.pkl", "wb") as f:
        pickle.dump(dict(q_table_x), f)
    with open("q_table_o.pkl", "wb") as f:
        pickle.dump(dict(q_table_o), f)
    print("Q-tables updated and saved.")

def play_human_vs_ai(human_player_preference="X", gui=True, train=True, alternate=False, game_number=1):
    """Play a game of Human vs AI with MCTS, with flexible player roles."""
    q_table_x, q_table_o = load_q_tables()
    env = EphemeralTicTacToeEnv()
    visualizer = Visualizer(gui=gui)
    obs = env.reset()
    
    # Determine human player role
    if alternate:
        human_player = "X" if game_number % 2 == 1 else "O"
    else:
        human_player = human_player_preference if env.game.current_player == human_player_preference else ("O" if env.game.current_player == "X" else "X")
    ai_player = "O" if human_player == "X" else "X"
    q_table_ai = q_table_o if ai_player == "O" else q_table_x
    
    action_history = [None, None, None]
    game_history = []  # (state, action, reward, next_state, done, player)
    done = False
    step_count = 0
    max_steps = 20

    print(f"Game {game_number}: Human as {human_player}, AI as {ai_player}")

    while not done and step_count < max_steps:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        state_key = hash_state(obs)

        if env.game.current_player == human_player:
            action = None
            while action not in legal_actions and not done:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        col, row = x // CELL_SIZE, y // CELL_SIZE
                        if 0 <= row < 3 and 0 <= col < 3:
                            action = row * 3 + col
                    visualizer.check_quit()
                if gui:
                    visualizer.draw_grid()
                    visualizer.draw_symbols(env)
                    pygame.display.flip()

            if action in legal_actions:
                next_obs, reward, done, info = env.step(action)
                game_history.append((obs.copy(), action, reward, next_obs.copy(), done, env.game.current_player))
                if gui:
                    visualizer.refresh(next_obs, reward, done, info, env, action_history)
                obs = next_obs
                step_count += 1

        else:
            action = mcts_search(env, obs, q_table_ai, iterations=1000)
            if action is None:
                print("No valid action from MCTS, ending game.")
                break
            next_obs, reward, done, info = env.step(action)
            game_history.append((obs.copy(), action, reward, next_obs.copy(), done, env.game.current_player))
            if gui:
                visualizer.refresh(next_obs, reward, done, info, env, action_history)
            obs = next_obs
            step_count += 1

    if done:
        if reward == 1:
            winner_player = info['player']
            winner = "Human" if winner_player == human_player else "AI"
            print(f"{winner} ({winner_player}) wins!")
        else:
            print("Draw!")

    if train:
        update_q_tables(game_history, q_table_x, q_table_o)

    if gui:
        while True:
            visualizer.check_quit()

    return game_history

if __name__ == "__main__":
    # Play multiple games with alternating roles
    num_games = 2  # Adjust as desired
    for game in range(1, num_games + 1):
        play_human_vs_ai(gui=True, train=True, alternate=True, game_number=game)