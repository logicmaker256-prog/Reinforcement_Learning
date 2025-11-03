# ===============================
# 📦 Step 1: ライブラリ準備
# ===============================
!pip install gymnasium[classic_control] torch matplotlib tqdm imageio -q

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import trange
import imageio
from IPython.display import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 🧠 Step 2: DQNネットワーク定義
# ===============================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ===============================
# ⚙️ Step 3: 環境とハイパーパラメータ
# ===============================
env = gym.make("CartPole-v1", render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
replay_buffer = deque(maxlen=50000)
batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
target_update_freq = 50
num_episodes = 1000

rewards_log = []
losses = []

# ===============================
# 🚀 Step 4: メイン学習ループ
# ===============================
for episode in trange(num_episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0
    done = False

    while not done:
        # ε-greedy 行動選択
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # 🎯 滑らか報酬補正（角度＆位置ペナルティ）
        angle_penalty = abs(next_state[0, 2].item()) * 2.0
        position_penalty = abs(next_state[0, 0].item()) * 0.5
        reward = reward - angle_penalty - position_penalty

        # 倒れたときの軽いペナルティ
        if done and total_reward < 195:
            reward -= 5.0

        # 経験をリプレイバッファに追加
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # 学習ステップ
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.BoolTensor(dones).to(device)

            q_values = policy_net(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + gamma * next_q * (~dones)

            loss = nn.functional.smooth_l1_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

    # ε減衰
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # ターゲットネット更新
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards_log.append(total_reward)

env.close()

# ===============================
# 📈 Step 5: グラフ描画（滑らか線付き）
# ===============================
window = 20
smoothed = np.convolve(rewards_log, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(rewards_log, alpha=0.3, label="raw")
plt.plot(smoothed, label=f"{window}-ep moving avg", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN (CartPole-v1) with Smooth Reward Function")
plt.legend()
plt.show()

# ===============================
# 🎬 Step 6: 学習済みモデルで動画生成
# ===============================
env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    frame = env.render()
    frames.append(frame)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = torch.argmax(q_values).item()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"🎉 再生エピソードのスコア: {total_reward:.1f}")

# mp4に保存
imageio.mimsave("cartpole_result.mp4", frames, fps=30)
print("🎥 動画を 'cartpole_result.mp4' として保存しました。")

# 動画をColab上で再生
from IPython.display import Video

# Colab上で動画を再生
Video("cartpole_result.mp4", embed=True)