import numpy as np

class EphemeralTicTacToeEnv:
    """Environment for Ephemeral Tic-Tac-Toe with expiring pieces."""

    def __init__(self, grid_size=3, lifespan_x=6, lifespan_o=6):
        self.grid_size = grid_size
        self.lifespan_x = lifespan_x
        self.lifespan_o = lifespan_o
        self.board = np.full((grid_size, grid_size), None, dtype=object)
        self.ages = np.zeros((grid_size, grid_size), dtype=int)
        self.owners = np.full((grid_size, grid_size), None, dtype=object)
        self.current_player = None
        self.move_count = 0

    def reset(self, starting_player=None):
        """Reset the environment and return the initial observation."""
        self.board.fill(None)
        self.ages.fill(0)
        self.owners.fill(None)
        self.current_player = np.random.choice(["X", "O"]) if starting_player is None else starting_player
        self.move_count = 0
        return self._get_observation()

    def step(self, action):
        """Take an action and return (observation, reward, done, info)."""
        row, col = divmod(action, self.grid_size)
        if self.board[row, col] is not None:
            return self._get_observation(), -0.1, False, {"reason": "illegal move"}

        self.move_count += 1
        self._update_ages()
        expired = self._expire_pieces()
        self.board[row, col] = self.current_player
        self.ages[row, col] = 0
        self.owners[row, col] = self.current_player

        if self._check_win(self.current_player):
            return self._get_observation(), 1, True, {"player": self.current_player}

        reward = 0
        near_win_count = self._count_near_wins(self.current_player)
        if near_win_count >= 2:
            reward += 0.5
        elif near_win_count == 1:
            reward += 0.3

        opponent = "O" if self.current_player == "X" else "X"
        opponent_near_wins_before = self._count_near_wins(opponent, before_move=True)
        opponent_near_wins_after = self._count_near_wins(opponent)
        if opponent_near_wins_before > opponent_near_wins_after:
            reward += 0.3

        if expired and near_win_count == 0:
            reward -= 0.05

        self.current_player = opponent
        if not self.get_legal_actions():
            return self._get_observation(), 0.2, True, {}

        return self._get_observation(), reward, False, {"player": self.current_player}

    def _get_observation(self):
        """Convert game state to a numerical observation."""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] == "X":
                    obs[i, j, 0] = 1
                elif self.board[i, j] == "O":
                    obs[i, j, 0] = -1
                obs[i, j, 1] = self.ages[i, j]
                if self.owners[i, j] == "X":
                    obs[i, j, 2] = 1
                elif self.owners[i, j] == "O":
                    obs[i, j, 2] = -1
        return obs

    def get_legal_actions(self):
        """Return a list of legal action indices."""
        return [i * self.grid_size + j for i, j in 
                [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.board[i, j] is None]]

    def _update_ages(self):
        """Increment the age of all pieces."""
        self.ages[self.board != None] += 1

    def _expire_pieces(self):
        """Remove pieces that exceed their lifespan, return True if any expired."""
        expired = False
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] is None:
                    continue
                lifespan = self.lifespan_x if self.owners[i, j] == "X" else self.lifespan_o
                if self.ages[i, j] >= lifespan:
                    self.board[i, j] = None
                    self.ages[i, j] = 0
                    self.owners[i, j] = None
                    expired = True
        return expired

    def _check_win(self, player):
        """Check if the specified player has won."""
        for i in range(self.grid_size):
            if sum(self.board[i, :] == player) >= 3:
                return True
            if sum(self.board[:, i] == player) >= 3:
                return True
        diag1 = sum(np.diag(self.board) == player)
        diag2 = sum(np.diag(np.fliplr(self.board)) == player)
        if diag1 >= 3 or diag2 >= 3:
            return True
        return False

    def _count_near_wins(self, player, before_move=False):
        """Count the number of near-win opportunities for the player."""
        board = self.board.copy() if before_move else self.board
        if before_move:
            temp_row, temp_col = np.where(self.ages == 0)
            if len(temp_row) > 0:
                board[temp_row[0], temp_col[0]] = None

        count = 0
        for i in range(self.grid_size):
            row = board[i, :]
            if sum(row == player) == 2 and sum(row == None) == 1:
                count += 1
            col = board[:, i]
            if sum(col == player) == 2 and sum(col == None) == 1:
                count += 1
        diag1 = np.diag(board)
        if sum(diag1 == player) == 2 and sum(diag1 == None) == 1:
            count += 1
        diag2 = np.diag(np.fliplr(board))
        if sum(diag2 == player) == 2 and sum(diag2 == None) == 1:
            count += 1
        return count