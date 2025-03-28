# Ephemeral Tic-Tac-Toe 🎲✨

Welcome to **Ephemeral Tic-Tac-Toe**, a twist on the classic game where pieces don’t stick around forever! Built with Python, Pygame, and a sprinkle of AI magic (Q-learning and Monte Carlo Tree Search), this project pits humans against a crafty AI in a race against time—or rather, a race against disappearing X’s and O’s. Ready to test your wits? Let’s dive in!

## What’s This All About? 🤔

In traditional Tic-Tac-Toe, you place your X or O, and it sits there smugly until someone wins or the board fills up. Boring, right? In *Ephemeral Tic-Tac-Toe*, every piece has a **lifespan** (6 turns by default), after which it vanishes into thin air! This adds a layer of strategy: not only do you need three in a row, but you’ve got to keep them alive long enough to claim victory. It’s like playing Tic-Tac-Toe with ghostly pieces that haunt the board for a fleeting moment.

The game comes with:
- A sleek Pygame GUI to watch the action unfold.
- A trainable AI opponent using Q-learning and MCTS (Monte Carlo Tree Search).
- Human vs. AI gameplay—can you outsmart the machine?

## Features 🌟

- **Ephemeral Mechanics**: X’s and O’s expire after 6 turns, shaking up the classic formula.
- **AI Smarts**: Trained with Q-learning and enhanced by MCTS for a challenging opponent.
- **Visual Flair**: Pygame-powered grid with lifespans shown as little dots (white for alive, gray for fading).
- **Human vs. AI Mode**: Play as X or O, with the option to alternate roles across games.
- **Training Mode**: Watch the AI learn over thousands of episodes—or train it yourself!
- **Customizable**: Tweak grid size, lifespans, and more in `config.py`.

## Getting Started 🚀

### Prerequisites
- **Python 3.x** (we recommend 3.8+ for maximum compatibility).
- **Pygame**: For the snazzy visuals (`pip install pygame`).
- **NumPy**: For all the matrix magic (`pip install numpy`).

### Installation
1. Clone this repo to your machine:
   ```bash
   git clone https://github.com/yourusername/ephemeral-tic-tac-toe.git
   cd ephemeral-tic-tac-toe
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (No `requirements.txt` yet? Just run `pip install pygame numpy`!)
3. Fire it up:
   - To play Human vs. AI:
     ```bash
     python main.py
     ```
   - To train the AI first:
     ```bash
     python train.py
     ```

### How to Play 🎮
- **Human Moves**: Click a cell on the 3x3 grid to place your X or O.
- **AI Moves**: The AI uses MCTS to pick its spot—watch it think!
- **Winning**: Get 3 in a row before your pieces expire. If the board fills up or no moves remain, it’s a draw.
- **GUI**: The bottom “console” shows the latest moves and rewards. Lifespan dots shrink as pieces age.

## Project Structure 🗂️

Here’s the lay of the land:
- **`config.py`**: Settings like grid size (3x3), lifespans (6 turns), and colors.
- **`game.py`**: Core game logic—board management, piece expiration, win checks.
- **`env.py`**: Environment wrapper for AI interaction, with observations and rewards.
- **`train.py`**: Q-learning training script to make the AI smarter.
- **`main.py`**: Human vs. AI gameplay with MCTS and optional training updates.
- **`visualization.py`**: Pygame goodness—grids, symbols, and endgame messages.

## Training the AI 🧠

Want a tougher opponent? Train the AI with `train.py`:
```bash
python train.py
```
- Runs 10,000 episodes by default (edit `episodes` in `train.py` to change).
- Saves Q-tables (`q_table_x.pkl` and `q_table_o.pkl`) for X and O.
- Watch it improve with epsilon-greedy exploration and a decaying learning rate!

Then jump into `main.py` to face your newly minted genius.

## Customization 🎨

Love tinkering? Edit `config.py` to:
- Change `GRID_SIZE` for a bigger board (e.g., 4x4).
- Adjust `LIFESPAN_X` and `LIFESPAN_O` for shorter or longer piece lives.
- Swap colors in `COLORS` or add image paths in `IMG_PATHS` for custom X’s and O’s.

## Contributing 🤝

Got ideas? Found a bug? We’d love your help!  
1. Fork the repo.
2. Create a branch (`git checkout -b feature/cool-idea`).
3. Commit your changes (`git commit -m "Added a wild new feature"`).
4. Push it (`git push origin feature/cool-idea`).
5. Open a Pull Request and tell us all about it!

## License 📜

This project is licensed under the MIT License—free to use, modify, and share. See [LICENSE](LICENSE) for details.

## Acknowledgments 🙌

- **xAI**: For inspiring AI-driven fun.
- **Pygame Community**: For making game dev a breeze.
- **You**: For checking out Ephemeral Tic-Tac-Toe!

---

Ready to play? Clone it, run it, and see if you can outlast the AI’s vanishing tricks. Happy gaming! 🎉

---
