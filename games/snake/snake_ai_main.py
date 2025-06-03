import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from games.snake.main import SnakeAndApple, rows, cols, DELAY


class MainGameEnv:
    """Wraps the original Tkinter game for RL."""

    def __init__(self, init_length=3):
        self.init_length = init_length
        self.action_space = 4  # left, right, up, down
        self.game = SnakeAndApple()
        self.reset()

    def reset(self):
        self.game.play_again()
        self.done = False
        self.game.window.update()
        return self._get_state()

    def _get_state(self):
        board = np.zeros((rows, cols), dtype=np.float32)
        for r, c in self.game.snake:
            board[r, c] = 1.0
        ar, ac = self.game.apple_cell
        board[ar, ac] = 2.0
        dir_map = {"Left": 0, "Right": 1, "Up": 2, "Down": 3}
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[dir_map[self.game.snake_heading]] = 1.0
        return np.concatenate([board.flatten(), dir_onehot])

    def step(self, action):
        key_map = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
        before = len(self.game.snake)
        self.game.update_snake(key_map[action])
        self.game.window.update()
        time.sleep(DELAY / 1000.0)
        reward = 0.0
        if self.game.crashed:
            self.done = True
            reward = -1.0
        elif len(self.game.snake) > before:
            reward = 1.0
        return self._get_state(), reward, self.done

    def close(self):
        self.game.window.destroy()


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def train(env, episodes=200, batch_size=64, gamma=0.99):
    state_dim = rows * cols + 4
    policy = DQN(state_dim)
    target = DQN(state_dim)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)
    epsilon = 1.0

    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        total_reward = 0.0
        while True:
            if random.random() < epsilon:
                action = random.randrange(env.action_space)
            else:
                with torch.no_grad():
                    qvals = policy(state)
                    action = int(torch.argmax(qvals))

            next_state, reward, done = env.step(action)
            next_state_t = torch.tensor(next_state, dtype=torch.float32)
            memory.append((state, action, reward, next_state_t, done))
            state = next_state_t
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                b_state, b_action, b_reward, b_next, b_done = zip(*batch)
                b_state = torch.stack(b_state)
                b_next = torch.stack(b_next)
                b_action = torch.tensor(b_action)
                b_reward = torch.tensor(b_reward, dtype=torch.float32)
                b_done = torch.tensor(b_done, dtype=torch.float32)

                q_values = policy(b_state).gather(1, b_action.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next = target(b_next).max(1)[0]
                    target_vals = b_reward + gamma * max_next * (1 - b_done)
                loss = nn.functional.mse_loss(q_values, target_vals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(0.1, epsilon * 0.995)
        if ep % 10 == 0:
            target.load_state_dict(policy.state_dict())
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}: score={len(env.game.snake)}, reward={total_reward}")

    return policy


def demo(env, policy, episodes=3):
    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        score = 0
        while True:
            with torch.no_grad():
                qvals = policy(state)
                action = int(torch.argmax(qvals))
            next_state, _, done = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32)
            score = len(env.game.snake)
            if done:
                env.game.display_gameover()
                env.game.window.update()
                print(f"Demo episode {ep+1}: score={score}")
                time.sleep(2)
                break


if __name__ == "__main__":
    env = MainGameEnv()
    policy = train(env)
    print("Training complete. Starting demo...")
    demo(env, policy)
    env.close()
