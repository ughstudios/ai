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

