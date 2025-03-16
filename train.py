import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from game import Game

# 设置随机种子以确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.manual_seed(SEED)

# 检查是否有可用的MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# DQN网络结构
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 输入层: 4x4网格，每个格子的值（使用对数表示）
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)  # 输出层: 4个动作（上、下、左、右）
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 游戏环境包装器
class GameEnv:
    def __init__(self):
        self.game = Game()
        self.action_space = 4  # 上、下、左、右
        
    def reset(self):
        self.game = Game()
        return self._get_state()
    
    def step(self, action):
        # 执行动作
        moved = False
        if action == 0:  # 上
            moved = self.game.move_up()
        elif action == 1:  # 下
            moved = self.game.move_down()
        elif action == 2:  # 左
            moved = self.game.move_left()
        elif action == 3:  # 右
            moved = self.game.move_right()
        
        # 如果移动有效，添加新的方块
        if moved:
            self.game.add_new_tile()
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._get_reward(moved)
        
        # 检查游戏是否结束
        done = self.game.is_game_over()
        
        return next_state, reward, done, {"score": self.game.score}
    
    def _get_state(self):
        # 将游戏网格转换为状态表示
        state = []
        for row in self.game.grid:
            for cell in row:
                # 使用对数表示，避免数值过大
                state.append(np.log2(cell) if cell > 0 else 0)
        return np.array(state, dtype=np.float32)
    
    def _get_reward(self, moved):
        # 如果移动无效，给予负奖励
        if not moved:
            return -10
        
        # 基础奖励为当前得分与上一步得分的差值
        reward = 0
        
        # 额外奖励：鼓励合并相同数字
        max_tile = max(max(row) for row in self.game.grid)
        empty_cells = sum(1 for i in range(4) for j in range(4) if self.game.grid[i][j] == 0)
        
        # 奖励最大方块值和空格数量
        reward += np.log2(max_tile) * 0.1
        reward += empty_cells * 0.5
        
        return reward

# DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # 创建Q网络和目标网络
        self.q_network = DQN().to(device)
        self.target_network = DQN().to(device)
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(self.buffer_size)
        
        # 训练步数计数器
        self.train_step = 0
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state, training=True):
        # ε-贪心策略选择动作
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def train(self):
        # 如果缓冲区中的样本不足，不进行训练
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从缓冲区中随机采样
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 定期更新目标网络
        self.train_step += 1
        if self.train_step % 100 == 0:
            self.update_target_network()
        
        return loss.item()

# 训练函数
def train_agent(episodes=1000, max_steps=10000, target_update=10, 
                save_interval=100, model_dir="models"):
    env = GameEnv()
    agent = DQNAgent(state_size=16, action_size=env.action_space)
    
    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 记录训练过程
    scores = []
    max_tiles = []
    losses = []
    
    for episode in tqdm(range(episodes), desc="训练进度"):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.memory.add(state, action, reward, next_state, done)
            
            # 训练智能体
            loss = agent.train()
            if loss:
                episode_loss += loss
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 记录本轮游戏的得分和最大方块
        scores.append(info["score"])
        max_tile = max(max(row) for row in env.game.grid)
        max_tiles.append(max_tile)
        losses.append(episode_loss / (step + 1) if step > 0 else 0)
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}, Score: {info['score']}, "
                  f"Max Tile: {max_tile}, Epsilon: {agent.epsilon:.4f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            torch.save(agent.q_network.state_dict(), 
                       os.path.join(model_dir, f"dqn_model_ep{episode+1}.pth"))
    
    # 保存最终模型
    torch.save(agent.q_network.state_dict(), os.path.join(model_dir, "dqn_model_final.pth"))
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('得分')
    plt.xlabel('回合')
    plt.ylabel('得分')
    
    plt.subplot(1, 3, 2)
    plt.plot(max_tiles)
    plt.title('最大方块')
    plt.xlabel('回合')
    plt.ylabel('最大方块值')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('损失')
    plt.xlabel('回合')
    plt.ylabel('平均损失')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curves.png"))
    plt.show()
    
    return agent

# 测试函数
def test_agent(model_path, episodes=10, render=False):
    env = GameEnv()
    agent = DQNAgent(state_size=16, action_size=env.action_space, epsilon=0)  # 测试时不使用探索
    
    # 加载模型
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()
    
    scores = []
    max_tiles = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state, training=False)
            
            # 执行动作
            next_state, _, done, info = env.step(action)
            
            state = next_state
            
            # 如果需要渲染，打印当前游戏状态
            if render:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"Episode: {episode+1}/{episodes}, Score: {info['score']}")
                for row in env.game.grid:
                    print(row)
                print("\n")
                import time
                time.sleep(0.1)
        
        # 记录本轮游戏的得分和最大方块
        scores.append(info["score"])
        max_tile = max(max(row) for row in env.game.grid)
        max_tiles.append(max_tile)
        
        print(f"Episode {episode+1}/{episodes}, Score: {info['score']}, Max Tile: {max_tile}")
    
    # 打印测试结果
    print(f"\n测试结果 ({episodes} 回合):")
    print(f"平均得分: {np.mean(scores):.2f}")
    print(f"最高得分: {np.max(scores)}")
    print(f"平均最大方块: {np.mean(max_tiles):.2f}")
    print(f"最大方块: {np.max(max_tiles)}")
    
    return scores, max_tiles

if __name__ == "__main__":
    # 训练智能体
    print("开始训练DQN智能体...")
    agent = train_agent(episodes=500, save_interval=50)
    
    # 测试智能体
    print("\n开始测试DQN智能体...")
    test_agent("models/dqn_model_final.pth", episodes=20, render=True)
