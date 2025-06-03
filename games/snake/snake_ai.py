import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class SnakeEnv:
    """A minimal Snake game environment for training."""

    def __init__(self, rows=10, cols=10, init_length=3):
        self.rows = rows
        self.cols = cols
        self.init_length = init_length
        self.action_space = 4  # left, right, up, down
        self.reset()

    def reset(self):
        self.direction = 1  # start moving right
        self.snake = [(i, 0) for i in range(self.init_length)]
        self.apple = self._random_cell()
        self.done = False
        return self._get_state()

    def _random_cell(self):
        cells = set((r, c) for r in range(self.rows) for c in range(self.cols))
        cells -= set(self.snake)
        return random.choice(list(cells))

    def _get_state(self):
        board = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in self.snake:
            board[r, c] = 1.0
        ar, ac = self.apple
        board[ar, ac] = 2.0
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.direction] = 1.0
        return np.concatenate([board.flatten(), dir_onehot])

    def step(self, action):
        if self.done:
            raise ValueError("Game over. Call reset().")

        # Forbid opposite direction
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposite[self.direction]:
            action = self.direction
        self.direction = action

        head = self.snake[-1]
        if action == 0:  # left
            new_head = (head[0] - 1, head[1])
        elif action == 1:  # right
            new_head = (head[0] + 1, head[1])
        elif action == 2:  # up
            new_head = (head[0], head[1] - 1)
        else:  # down
            new_head = (head[0], head[1] + 1)

        reward = 0.0
        if (
            new_head[0] < 0
            or new_head[0] >= self.rows
            or new_head[1] < 0
            or new_head[1] >= self.cols
            or new_head in self.snake
        ):
            self.done = True
            reward = -1.0
            return self._get_state(), reward, self.done

        self.snake.append(new_head)
        if new_head == self.apple:
            reward = 1.0
            self.apple = self._random_cell()
        else:
            self.snake.pop(0)

        return self._get_state(), reward, self.done


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


def train(env, episodes=500, batch_size=64, gamma=0.99):
    state_dim = env.rows * env.cols + 4
    policy = DQN(state_dim)
    target = DQN(state_dim)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)
    epsilon = 1.0

    for ep in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
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
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}: score={len(env.snake)}, reward={total_reward}")

    return policy


def demo(env, policy, episodes=5):
    for ep in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        score = 0
        while True:
            with torch.no_grad():
                qvals = policy(state)
                action = int(torch.argmax(qvals))
            next_state, _, done = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32)
            score = len(env.snake)
            if done:
                print(f"Demo episode {ep+1}: score={score}")
                break


if __name__ == "__main__":
    env = SnakeEnv()
    trained_policy = train(env)
    demo(env, trained_policy)
