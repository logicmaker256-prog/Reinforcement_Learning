# ================================
# 🚗 MountainCar Double DQN (安定完全版)
# ================================
!pip install gymnasium==0.29.1 moviepy > /dev/null

import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from IPython.display import Video, display

# --- Double DQNモデル定義 ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.fc(x)

# --- ε-greedyで行動選択 ---
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(torch.FloatTensor(state))).item()

# --- ターゲットネット更新 ---
def update_target():
    target_net.load_state_dict(policy_net.state_dict())

# --- MountainCar環境 ---
env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- ネットワーク・オプティマイザ ---
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
update_target()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=50000)

# --- ハイパーパラメータ ---
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.990
num_episodes = 1000
target_update = 10
rewards_log = []

# --- 学習ループ ---
for ep in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- 報酬 shaping（登るほど報酬） ---
        position, velocity = next_state
        reward = abs(position - (-0.5))
        if position >= 0.5:
            reward += 100  # ゴールボーナス

        # 経験を保存
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # --- Double DQN の更新 ---
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # (1) 現在のQ値
            q_values = policy_net(states).gather(1, actions).squeeze(1)

            # (2) Double DQNターゲット計算
            next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + gamma * next_q_values * (1 - dones)

            # (3) 損失計算・更新
            loss = nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- ε減衰 ---
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_log.append(total_reward)

    if ep % target_update == 0:
        update_target()

    print(f"Ep {ep:3d}  Reward={total_reward:7.2f}  Eps={epsilon:.3f}")

env.close()

# --- 学習曲線表示 ---
plt.plot(rewards_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("🏔️ Double DQN Training Reward (MountainCar-v0)")
plt.grid()
plt.show()

# ==========================
# 🎥 学習済みモデルで動画撮影
# ==========================
video_path = "mountaincar_double_dqn.mp4"
env = gym.make("MountainCar-v0", render_mode="rgb_array")
frames = []

state, _ = env.reset()
done = False
while not done:
    frame = env.render()
    frames.append(frame)
    action = select_action(state, epsilon=0.0)  # 完全学習済みで実行
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
imageio.mimsave(video_path, frames, fps=30)

# --- Colab上で動画再生 ---
display(Video(video_path, embed=True))