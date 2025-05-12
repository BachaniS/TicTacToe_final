import pygame

class Visualizer:
    def __init__(self, gui=True):
        self.gui = gui
        self.cell_size = 200
        self.grid_size = 3
        self.width = self.grid_size * self.cell_size
        self.height = self.width + 120
        self.colors = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "GRAY": (200, 200, 200),
            "DARK_GRAY": (50, 50, 50),
            "PLAYER1": (100, 149, 237),  # X
            "PLAYER2": (255, 99, 71),    # O
        }
        self.screen = None
        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Ephemeral Tic-Tac-Toe")
            self.font = pygame.font.SysFont('arial', 48)
            self.small_font = pygame.font.SysFont('arial', 30)
            self.clock = pygame.time.Clock()
            self.screen.fill(self.colors["WHITE"])  # Initialize with white background
            pygame.display.flip()

    def draw_grid(self):
        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, self.colors["GRAY"], (i * self.cell_size, 0), (i * self.cell_size, self.width), 5)
            pygame.draw.line(self.screen, self.colors["GRAY"], (0, i * self.cell_size), (self.width, i * self.cell_size), 5)

    def draw_symbols(self, env):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                symbol = env.board[i, j]
                if symbol:
                    age = env.ages[i, j]
                    lifespan = env.lifespan_x if symbol == 'X' else env.lifespan_o
                    center_x = j * self.cell_size + self.cell_size // 2
                    center_y = i * self.cell_size + self.cell_size // 2

                    if age >= lifespan:  # Expired piece: draw a cross
                        pygame.draw.line(self.screen, self.colors["BLACK"], 
                                        (center_x - 30, center_y - 30), (center_x + 30, center_y + 30), 5)
                        pygame.draw.line(self.screen, self.colors["BLACK"], 
                                        (center_x - 30, center_y + 30), (center_x + 30, center_y - 30), 5)
                    else:  # Active piece: draw symbol and lifespan circles
                        alpha = max(255 - (255 * age // lifespan), 50)
                        color = self.colors["PLAYER1"] if symbol == "X" else self.colors["PLAYER2"]
                        symbol_surface = self.font.render(symbol, True, color)
                        symbol_surface.set_alpha(alpha)
                        rect = symbol_surface.get_rect(center=(center_x, center_y))
                        self.screen.blit(symbol_surface, rect)

                        # Draw lifespan circles (up to 3, based on remaining lifespan)
                        remaining_life = lifespan - age
                        circles = min(3, max(0, int(remaining_life / (lifespan / 3.0))))
                        for c in range(circles):
                            pygame.draw.circle(self.screen, color, 
                                             (center_x - 30 + c * 30, center_y - 50), 10)

                    # Draw age below the symbol
                    age_surface = self.small_font.render(str(age), True, self.colors["BLACK"])
                    age_rect = age_surface.get_rect(center=(center_x, center_y + 40))
                    self.screen.blit(age_surface, age_rect)

    def display_turn_info(self, env):
        info_surface = self.small_font.render(f"Current Turn: {env.current_player}", True, self.colors["DARK_GRAY"])
        rect = info_surface.get_rect(center=(self.width // 2, self.height - 60))
        self.screen.blit(info_surface, rect)

    def refresh(self, obs, reward, done, info, env, action_history):
        if not self.gui:
            return
        self.screen.fill(self.colors["WHITE"])  # Clear screen with white background
        self.draw_grid()
        self.draw_symbols(env)
        self.display_turn_info(env)

        if done:
            if reward == 1:
                winner_text = f"Winner: {info['player']}!"
            else:
                winner_text = "It's a Draw!"
            winner_surface = self.font.render(winner_text, True, self.colors["BLACK"])
            winner_rect = winner_surface.get_rect(center=(self.width // 2, self.height // 2))
            pygame.draw.rect(self.screen, self.colors["WHITE"], winner_rect.inflate(20, 20))
            self.screen.blit(winner_surface, winner_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def check_quit(self):
        if not self.gui:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

    def close(self):
        if self.gui:
            pygame.quit()