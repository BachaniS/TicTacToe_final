# env.py
import numpy as np
from game import EphemeralTicTacToe

class EphemeralTicTacToeEnv:
    """Environment wrapper for Ephemeral Tic-Tac-Toe."""

    def __init__(self, grid_size=3, lifespan_x=6, lifespan_o=6):
        self.game = EphemeralTicTacToe(grid_size, lifespan_x, lifespan_o)
        self.grid_size = grid_size

    def reset(self):
        """Reset the environment and return the initial observation."""
        state = self.game.reset()
        return self._get_observation(state)

    def step(self, action):
        """Take an action and return (observation, reward, done, info)."""
        row, col = divmod(action, self.grid_size)
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            return self._get_observation(self.game.get_state()), -0.1, False, {"reason": "illegal move"}
        
        current_player_before = self.game.current_player  # Capture player before stepping
        state, reward, done = self.game.step((row, col))
        info = {"action": (row, col), "player": current_player_before}
        return self._get_observation(state), reward, done, info

    def _get_observation(self, state):
        """Convert game state to a numerical observation."""
        board, ages, _, owners = state
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if board[i, j] == "X":
                    obs[i, j, 0] = 1
                elif board[i, j] == "O":
                    obs[i, j, 0] = -1
                obs[i, j, 1] = ages[i, j]
                if owners[i, j] == "X":
                    obs[i, j, 2] = 1
                elif owners[i, j] == "O":
                    obs[i, j, 2] = -1
        return obs

    def get_legal_actions(self):
        """Return a list of legal action indices."""
        return [i * self.grid_size + j for i, j in self.game.get_legal_actions()]