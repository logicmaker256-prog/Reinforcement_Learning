import numpy as np
import matplotlib.pyplot as plt
import random
import time
from IPython.display import Image
import imageio.v2 as imageio
import os

# ======== 環境定義（障害物あり） ========
class GridWorld:
    def __init__(self, size=5, obstacles=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        # 障害物セット
        self.obstacles = obstacles if obstacles else set()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        next_x, next_y = x, y

        if action == 0:   # 上
            next_x = max(x - 1, 0)
        elif action == 1: # 下
            next_x = min(x + 1, self.size - 1)
        elif action == 2: # 左
            next_y = max(y - 1, 0)
        elif action == 3: # 右
            next_y = min(y + 1, self.size - 1)

        # 障害物にぶつかったらその場に留まる
        if (next_x, next_y) in self.obstacles:
            next_x, next_y = x, y

        self.state = (next_x, next_y)

        # 報酬設定
        if self.state == self.goal:
            return self.state, 100, True
        else:
            return self.state, -1, False

    def render_image(self, path=None):
        """画像出力（障害物つき）"""
        grid = np.ones((self.size, self.size, 3))

        # 障害物（黒）
        for (x, y) in self.obstacles:
            grid[x, y] = [0, 0, 0]

        # 通過経路（灰）
        if path:
            for (x, y) in path:
                if (x, y) not in self.obstacles:
                    grid[x, y] = [0.8, 0.8, 0.8]

        # スタート・ゴール・エージェント
        sx, sy = self.start
        gx, gy = self.goal
        ax, ay = self.state
        grid[sx, sy] = [0.3, 0.7, 1.0]   # 青: Start
        grid[gx, gy] = [1.0, 0.4, 0.4]   # 赤: Goal
        grid[ax, ay] = [0.2, 0.9, 0.2]   # 緑: Agent
        return grid


# ======== 環境セットアップ ========
obstacles = {(1, 1), (2, 1), (3, 3), (1, 3)}
env = GridWorld(size=5, obstacles=obstacles)

# ======== Q学習 ========
q_table = np.zeros((env.size, env.size, 4))
alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 1500
rewards = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):
        x, y = state
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[x, y])

        next_state, reward, done = env.step(action)
        nx, ny = next_state

        q_table[x, y, action] += alpha * (
            reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )

        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)

# ======== 学習曲線 ========
plt.figure(figsize=(6, 3))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve (GridWorld with Obstacles)")
plt.grid()
plt.show()

# ======== 経路アニメーション（GIF出力） ========
frames = []
state = env.reset()
path = [state]
os.makedirs("frames", exist_ok=True)

for step in range(30):
    img = env.render_image(path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Step {step}")
    filename = f"frames/frame_{step:03d}.png"
    plt.savefig(filename)
    plt.close()
    frames.append(imageio.imread(filename))

    x, y = state
    action = np.argmax(q_table[x, y])
    next_state, reward, done = env.step(action)
    state = next_state
    path.append(state)
    if done:
        break

# 最終フレーム
img = env.render_image(path)
plt.imshow(img)
plt.axis("off")
plt.title("Goal!")
filename = f"frames/frame_goal.png"
plt.savefig(filename)
plt.close()
frames.append(imageio.imread(filename))

# ======== GIF保存 ========
gif_path = "/content/gridworld_obstacle.gif"
imageio.mimsave(gif_path, frames, duration=0.4)
print(f"🎥 GIF保存完了: {gif_path}")

# Colab上で表示
Image(filename=gif_path)