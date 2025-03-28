# visualization.py
import pygame
from config import WIDTH, HEIGHT, CELL_SIZE, GRID_SIZE, COLORS, FPS, IMG_PATHS

class Visualizer:
    """Handles visualization for Ephemeral Tic-Tac-Toe."""

    def __init__(self, gui=True):
        self.gui = gui
        self.screen = None
        self.clock = None
        self.images = {}
        if gui:
            self._initialize_pygame()

    def _initialize_pygame(self):
        """Initialize Pygame and load resources."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ephemeral Tic-Tac-Toe")
        self.clock = pygame.time.Clock()
        self._load_images()

    def _load_images(self):
        """Load and scale images if provided."""
        for key, path in IMG_PATHS.items():
            if path:
                try:
                    img = pygame.image.load(path)
                    if key in ["X", "O"]:
                        img = pygame.transform.scale(img, (CELL_SIZE // 2, CELL_SIZE // 2))
                    elif key == "BACKGROUND":
                        img = pygame.transform.scale(img, (WIDTH, WIDTH))
                    self.images[key] = img
                except pygame.error as e:
                    print(f"Failed to load {key} image: {e}")
                    self.images[key] = None

    def draw_grid(self):
        """Draw the game grid and console area."""
        self.screen.fill(COLORS["BLACK"])
        if "BACKGROUND" in self.images and self.images["BACKGROUND"]:
            self.screen.blit(self.images["BACKGROUND"], (0, 0))
        else:
            pygame.draw.rect(self.screen, COLORS["WHITE"], (0, 0, WIDTH, WIDTH))
        for x in range(0, WIDTH, CELL_SIZE):
            for y in range(0, WIDTH, CELL_SIZE):
                pygame.draw.rect(self.screen, COLORS["BLACK"], (x, y, CELL_SIZE, CELL_SIZE), 1)
        pygame.draw.rect(self.screen, COLORS["GRAY"], (0, WIDTH, WIDTH, HEIGHT - WIDTH))

    def draw_symbols(self, env):
        """Draw game symbols and lifespans."""
        board, ages, _, owners = env.game.get_state()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x, y = col * CELL_SIZE, row * CELL_SIZE
                center_x, center_y = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                if board[row, col] == "X":
                    if "X" in self.images and self.images["X"]:
                        self.screen.blit(self.images["X"], (center_x - CELL_SIZE // 4, center_y - CELL_SIZE // 4))
                    else:
                        pygame.draw.line(self.screen, COLORS["PLAYER1"], (x + 40, y + 40), (x + CELL_SIZE - 40, y + CELL_SIZE - 40), 10)
                        pygame.draw.line(self.screen, COLORS["PLAYER1"], (x + CELL_SIZE - 40, y + 40), (x + 40, y + CELL_SIZE - 40), 10)
                elif board[row, col] == "O":
                    if "O" in self.images and self.images["O"]:
                        self.screen.blit(self.images["O"], (center_x - CELL_SIZE // 4, center_y - CELL_SIZE // 4))
                    else:
                        pygame.draw.circle(self.screen, COLORS["PLAYER2"], (center_x, center_y), CELL_SIZE // 3, 10)
                
                if board[row, col] is not None:
                    lifespan = env.game.lifespan_x if owners[row, col] == "X" else env.game.lifespan_o
                    remaining_life = lifespan - ages[row, col]
                    for i in range(lifespan):
                        circle_pos = (x + 10 + i * 15, y + 20)
                        color = COLORS["WHITE"] if i < remaining_life else COLORS["DARK_GRAY"]
                        pygame.draw.circle(self.screen, color, circle_pos, 6, 0 if i < remaining_life else 2)

    def display_end_message(self, message):
        """Display the game end message."""
        font = pygame.font.Font(None, 100)
        text_surface = font.render(message, True, COLORS["DARK_GRAY"])
        text_rect = text_surface.get_rect(center=(WIDTH // 2, WIDTH // 2))
        self.screen.blit(text_surface, text_rect)

    def refresh(self, obs, reward, done, info, env, action_history, delay=1.0):
        """Update the display with the current game state."""
        if not self.gui:
            return
        
        # Use the player from the step info
        action = info.get("action", (0, 0))
        player = info.get("player", None)
        result = f"Player {player} moved to {action}, Reward: {reward}" if player is not None else "Illegal move"
        
        if None in action_history:
            action_history[action_history.index(None)] = result
        else:
            action_history.pop(0)
            action_history.append(result)
        
        self.draw_grid()
        self.draw_symbols(env)
        
        if done:
            if reward == 1:
                end_message = f"{player} Wins!"
            else:
                end_message = "Draw!"
            self.display_end_message(end_message)

        font = pygame.font.Font(None, 30)
        self.screen.blit(font.render("Actions", True, COLORS["BLACK"]), (10, WIDTH + 10))
        font = pygame.font.Font(None, 24)
        y_offset = WIDTH + 40
        for result in action_history:
            if result:
                self.screen.blit(font.render(result, True, COLORS["BLACK"]), (10, y_offset))
                y_offset += 30

        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.time.wait(int(delay * 1000))

    def check_quit(self):
        """Check for quit events and handle them."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit