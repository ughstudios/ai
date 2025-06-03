# AI Experiments

This repository collects resources and small projects. The `games/` directory
contains open-source games for experimentation. Currently it includes a Snake
game located in `games/snake/`.

The `games/snake` folder now also includes `snake_ai.py`, which trains a
simple neural network using a DQN approach to play Snake automatically.
Run the following to train and watch a short demo. `matplotlib` is
required to display the board during the demo:

```bash
pip install torch numpy matplotlib
python games/snake/snake_ai.py
```

Training may take a few minutes on a CPU. Once it completes, a window will
open showing the agent playing several games automatically. Progress prints
every 50 episodes; when you see "Training complete. Starting demo..." the
board window should remain open until you close it. If no window appears,
ensure that `matplotlib` is installed and that your environment is able to
display GUI windows.
