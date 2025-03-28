# config.py
GRID_SIZE = 3
CELL_SIZE = 200  # Adjusted for WIDTH = 600
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = WIDTH + 120  # Extra space for console
LIFESPAN_X = 6
LIFESPAN_O = 6
FPS = 60
SLEEP_TIME = 1.0

COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (200, 200, 200),
    "DARK_GRAY": (50, 50, 50),
    "PLAYER1": (100, 149, 237),  # X
    "PLAYER2": (255, 99, 71),    # O
}

IMG_PATHS = {
    "X": None,  # Update with actual paths if needed
    "O": None,
    "BACKGROUND": None,
}