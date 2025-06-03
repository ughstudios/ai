# Snake-And-Apple

This folder contains a Python-based interactive Snake game.

## Running the game
```bash
python main.py
```

The original project included images for screenshots and an animated
preview. These were removed to keep the repository lightweight.

## Controls
1. The game begins with a snake of length 3 waiting for user input
2. Keyboard Up, Down, Right, and Left are used to navigate
3. The result of the game is displayed at the end of the game
4. Click anywhere on the result screen to play again

## Author
[Aqeel Anwar](https://www.prism.gatech.edu/~manwar8)

## Training an AI Player
The file `snake_ai.py` implements a DQN agent that learns to play Snake.
Execute the script to start training and see a demo. `matplotlib` is
required for visualizing the board:

```bash
pip install torch numpy matplotlib
python snake_ai.py
```


Note that the AI player does not use the Tkinter game defined in `main.py`.
`snake_ai.py` implements a streamlined environment rendered with `matplotlib`
so training can run without a GUI and with minimal overhead.

### Training with the Tkinter GUI

If you prefer to watch training inside the original Tkinter window, run
`snake_ai_tk.py` instead. The agent uses the same DQN logic but renders each
step in real time using the `main.py` board:

```bash
pip install torch numpy
python snake_ai_tk.py
```

### AI demo in the original game

`snake_ai_main.py` runs training and inference using the actual Tkinter board
from `main.py`. This lets you watch the agent play with the normal game over
screen:

```bash
pip install torch numpy pillow
python snake_ai_main.py
```
