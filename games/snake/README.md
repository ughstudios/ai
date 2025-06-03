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

Training might take a short while. After it finishes, the script opens a matplotlib window showing the trained snake play several games. If you do not see any window, check your `matplotlib` installation and GUI support.
Progress is printed every 50 episodes. When "Training complete. Starting demo..." appears, the window will remain open until you close it.
