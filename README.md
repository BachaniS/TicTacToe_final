🎲 Ephemeral Tic-Tac-Toe: The Game Where Pieces Vanish! ✨
Welcome to Ephemeral Tic-Tac-Toe, a wild twist on the classic game where X's and O's have a limited lifespan and poof out of existence! 🕒 Battle a crafty AI, watch pieces fade, and chase victory before your moves expire. Ready to dive into this fast-paced, brain-tickling challenge? Let’s go! 🚀
🎮 What's So Special?
In this 3x3 Tic-Tac-Toe adventure:

Pieces Expire: X and O pieces vanish after 6 turns, so plan fast! ⏳
Win or Bust: Get three in a row (row, column, diagonal) before your pieces disappear.
Fancy Visuals: Watch pieces fade, lifespan circles shrink, and expired spots turn into crosses. 😎
Human vs. AI: You (X) face off against a Q-learning AI (O) that learns from your moves! 🧠
Interactive GUI: Click to place pieces, see the board update instantly, and enjoy a lively display. 🖱️

🛠️ Get Started in a Snap!
What You Need

Python 3.8+ 🐍
Libraries: pygame (for the cool GUI) and numpy (for board magic)
Install them like a pro:pip install pygame numpy



Files in Your Arsenal

env.py: The game’s heart, handling the board and expiration rules.
visualization.py: The artist, painting the board with fading pieces and circles. 🎨
train.py: The AI’s gym, training it to be a worthy opponent.
main.py: The arena where you battle the AI.

Step 1: Train the AI 🏋️‍♂️

Save all files in one folder.
Run train.py to teach the AI some tricks:python train.py


Tweak train.py for fun:
GUI = True: Watch training live (warning: it’s hypnotic!).
EPISODES = 50000: More episodes = smarter AI.
VISUALIZE_EVERY = 1: Show every training game (set higher to speed up).


This creates q_table_agent.pkl, the AI’s brain.

Step 2: Play the Game! 🎉

Run main.py to challenge the AI:python main.py


Customize main.py:
GUI = True: Enjoy the graphical board (default).
NRUNS = 1: Play multiple games by increasing this.


How to Play:
GUI Mode: Click a cell to place your X. The AI responds with an O. 🖱️
Console Mode (set GUI = False): Enter row col (e.g., 0 1).
Close the window to quit or keep playing for glory!



🌟 Cool Visuals to Love
The GUI is where the magic happens! ✨

Fading Pieces: X (blue) and O (red) fade as they age, like ghosts! 👻
Lifespan Circles: Up to 3 circles above each piece show how long it’s got left (~2 turns per circle for lifespan=6). They vanish as time runs out. ⭕
Expired Pieces: When a piece hits its limit (age ≥ 6), it turns into a bold black cross. ❌
Age Display: Each piece or cross shows its age below, so you know who’s on borrowed time.
Turn Info: The bottom says whose turn it is (X or O).
Game End: See “Winner: X/O!” or “It’s a Draw!” in big, bold text. 🏆

🐛 Oops, Something Went Wrong?
No worries, we’ve got you! 😄

Black Screen? 😵
Update Pygame: pip install pygame --upgrade.
Ensure GUI = True in main.py or train.py.
Check your display supports 600x720 resolution.


Missing Q-Table? 🤔
Run train.py first to generate q_table_agent.pkl.


Can’t Move? 😩
In GUI, click inside the 3x3 grid only.
In console, use valid coordinates (0-2 for row and column).


Training Too Slow? 🐢
Set GUI = False or increase VISUALIZE_EVERY in train.py.



🎈 Make It Your Own!
Want to crank up the fun? Try these:

Change lifespan_x or lifespan_o in env.py to make pieces last longer or shorter. 🕰️
Tweak MOVE_DELAY (500ms) or END_GAME_DELAY (1000ms) in main.py for faster/slower pacing.
Experiment with EPISODES in train.py to make the AI a genius or a rookie. 🧠
Add your own flair to visualization.py (e.g., new colors, fonts). 🎨

🚀 Jump In and Play!
Ephemeral Tic-Tac-Toe is a blast of strategy and surprises. Will you outsmart the AI before your pieces vanish? Grab your mouse, fire up the game, and let’s see who rules the board! 🏅
Made with 💖 for game lovers and code tinkerers. No license, just play and share the fun!
