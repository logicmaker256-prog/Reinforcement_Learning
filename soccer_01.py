# ============================================================
# ⚽ 攻撃型サッカー環境（敵キャラ入り・Colab完全動作版）
# ============================================================
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# 1. 環境定義
# ============================================================
class SoccerEnv(gym.Env):
    def __init__(self):
        super(SoccerEnv, self).__init__()
        self.width = 10
        self.height = 6
        self.goal_x = self.width - 1
        self.goal_y = self.height // 2
        self.action_space = gym.spaces.Discrete(5)  # 上下左右＋その場
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.player = np.array([1, self.height // 2])
        self.ball = np.array([2, self.height // 2])
        self.enemy = np.array([7, self.height // 2])
        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.player[0] / self.width,
            self.player[1] / self.height,
            self.ball[0] / self.width,
            self.ball[1] / self.height,
            self.enemy[0] / self.width,
            self.enemy[1] / self.height
        ], dtype=np.float32)

    def step(self, action):
        move = {0: np.array([0, -1]), 1: np.array([0, 1]),
                2: np.array([-1, 0]), 3: np.array([1, 0]), 4: np.array([0, 0])}
        self.player = np.clip(self.player + move[action], [0, 0], [self.width - 1, self.height - 1])

        # ボールが近いなら一緒に動く
        if np.linalg.norm(self.player - self.ball) < 1.1:
            self.ball = np.clip(self.ball + move[action], [0, 0], [self.width - 1, self.height - 1])

        # 敵キャラの単純な追跡行動
        diff = self.player - self.enemy
        self.enemy += np.sign(diff)
        self.enemy = np.clip(self.enemy, [0, 0], [self.width - 1, self.height - 1])

        done = False
        reward = -1  # 時間ペナルティ

        # ボールがゴールに入った
        if self.ball[0] == self.goal_x and self.ball[1] == self.goal_y:
            reward += 500
            done = True

        # 敵がボールを奪う
        if np.array_equal(self.enemy, self.ball):
            reward -= 300
            done = True

        # ゴール方向に進むほど報酬
        reward += (self.ball[0] - 2) * 0.5

        return self._get_obs(), reward, done, {}

    def render(self, mode="rgb_array"):
        img = np.ones((self.height, self.width, 3))
        img[:] = [0.9, 0.9, 0.9]
        img[self.goal_y, self.goal_x] = [0, 1, 0]       # ゴール
        img[self.ball[1], self.ball[0]] = [1, 0.5, 0]   # ボール
        img[self.player[1], self.player[0]] = [0, 0, 1] # プレイヤー
        img[self.enemy[1], self.enemy[0]] = [1, 0, 0]   # 敵
        return img


# ============================================================
# 2. DQNネットワーク定義
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 3. 学習設定
# ============================================================
env = SoccerEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# メモリ
memory = []
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.998
batch_size = 64

# ============================================================
# 4. 学習ループ
# ============================================================
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num_episodes = 1000
rewards_per_episode = []

for e in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(100):
        if random.random() < epsilon:
            action = random.randrange(action_size)
        else:
            with torch.no_grad():
                action = torch.argmax(policy_net(torch.FloatTensor(state))).item()

        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        if len(memory) > 50000:
            memory.pop(0)

        state = next_state
        total_reward += reward
        replay()
        if done:
            break
    rewards_per_episode.append(total_reward)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if e % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {e}  Reward: {total_reward:.2f}  Epsilon: {epsilon:.3f}")

plt.plot(rewards_per_episode)
plt.title("Episode Rewards")
plt.show()


# ============================================================
# 5. 学習結果の動画生成
# ============================================================
frames = []
state = env.reset()
for _ in range(100):
    frames.append({
        "player": env.player.copy(),
        "ball": env.ball.copy(),
        "enemy": env.enemy.copy()
    })
    with torch.no_grad():
        action = torch.argmax(policy_net(torch.FloatTensor(state))).item()
    state, _, done, _ = env.step(action)
    if done:
        break

# ============================================================
# 6. アニメーション再生（Colab対応）
# ============================================================
fig, ax = plt.subplots()
ax.set_xlim(0, env.width)
ax.set_ylim(0, env.height)
line_player, = ax.plot([], [], 'bo', markersize=12)
line_ball, = ax.plot([], [], 'yo', markersize=10)
line_enemy, = ax.plot([], [], 'ro', markersize=12)
ax.invert_yaxis()

def init():
    line_player.set_data([], [])
    line_ball.set_data([], [])
    line_enemy.set_data([], [])
    return line_player, line_ball, line_enemy

def update(frame):
    player_x, player_y = frames[frame]["player"]
    ball_x, ball_y = frames[frame]["ball"]
    enemy_x, enemy_y = frames[frame]["enemy"]
    line_player.set_data([player_x], [player_y])
    line_ball.set_data([ball_x], [ball_y])
    line_enemy.set_data([enemy_x], [enemy_y])
    return line_player, line_ball, line_enemy

anim = animation.FuncAnimation(fig, update, frames=len(frames),
                               init_func=init, blit=False, interval=150)
display(HTML(anim.to_jshtml()))