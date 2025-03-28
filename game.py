# game.py
import numpy as np
import random

class EphemeralTicTacToe:
    """Ephemeral Tic-Tac-Toe game with expiring pieces."""
    
    def __init__(self, grid_size=3, lifespan_x=6, lifespan_o=6):
        self.grid_size = grid_size
        self.lifespan_x = lifespan_x
        self.lifespan_o = lifespan_o
        self.reset()

    def reset(self, starting_player=None):
        """Reset the game state."""
        self.board = np.full((self.grid_size, self.grid_size), None, dtype=object)
        self.ages = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.owners = np.full((self.grid_size, self.grid_size), None, dtype=object)
        self.current_player = random.choice(["X", "O"]) if starting_player is None else starting_player
        self.move_count = 0
        return self.get_state()

    def get_state(self):
        """Return the current game state."""
        return (self.board.copy(), self.ages.copy(), self.current_player, self.owners.copy())

    def step(self, action):
        """Execute a move and return (state, reward, done)."""
        row, col = action
        if self.board[row, col] is not None:
            return self.get_state(), -0.1, False

        self.move_count += 1
        self._update_ages()
        self._expire_pieces()
        self.board[row, col] = self.current_player
        self.ages[row, col] = 0
        self.owners[row, col] = self.current_player

        if self._check_win(self.current_player):
            return self.get_state(), 1, True

        self.current_player = "O" if self.current_player == "X" else "X"
        if not self.get_legal_actions():
            return self.get_state(), 0, True

        return self.get_state(), 0, False

    def _update_ages(self):
        """Increment the age of all pieces."""
        self.ages[self.board != None] += 1

    def _expire_pieces(self):
        """Remove pieces that exceed their lifespan."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] is None:
                    continue
                lifespan = self.lifespan_x if self.owners[i, j] == "X" else self.lifespan_o
                if self.ages[i, j] >= lifespan:
                    self.board[i, j] = None
                    self.ages[i, j] = 0
                    self.owners[i, j] = None

    def _check_win(self, player):
        """Check if the specified player has won."""
        board = self.board
        # Check rows
        for i in range(self.grid_size):
            row_count = sum(board[i, :] == player)
            if row_count >= 3:
                return True
        
        # Check columns
        for j in range(self.grid_size):
            col_count = sum(board[:, j] == player)
            if col_count >= 3:
                return True
        
        # Check diagonals
        diag1_count = sum(np.diag(board) == player)
        diag2_count = sum(np.diag(np.fliplr(board)) == player)
        
        if diag1_count >= 3 or diag2_count >= 3:
            return True
        
        return False

    def get_legal_actions(self):
        """Return a list of legal moves."""
        return [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) 
                if self.board[i, j] is None]