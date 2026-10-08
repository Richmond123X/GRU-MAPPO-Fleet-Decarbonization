"""QMIX算法实现
基于价值分解的多智能体强化学习算法，适用于海运脱碳决策优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, namedtuple
import copy
import random

from ship import Ship


# 经验回放缓冲区的数据结构
Transition = namedtuple('Transition', 
                       ['state', 'actions', 'rewards', 'next_state', 'dones', 'global_state', 'next_global_state'])


class QNetwork(nn.Module):
    """单智能体Q网络 - 为每艘船估计Q值"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 构建多层神经网络
        layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            obs: 观测 [batch_size, obs_dim] 或 [batch_size, num_agents, obs_dim]
        Returns:
            q_values: Q值 [batch_size, action_dim] 或 [batch_size, num_agents, action_dim]
        """
        original_shape = obs.shape
        if len(original_shape) == 3:  # [batch_size, num_agents, obs_dim]
            batch_size, num_agents, obs_dim = original_shape
            obs = obs.view(-1, obs_dim)
            q_values = self.network(obs)
            q_values = q_values.view(batch_size, num_agents, self.action_dim)
        else:  # [batch_size, obs_dim]
            q_values = self.network(obs)
        
        return q_values


class MixingNetwork(nn.Module):
    """混合网络 - 将个体Q值混合为全局Q值"""
    
    def __init__(self, num_agents: int, global_state_dim: int, hidden_dim: int = 256, hypernet_layers: int = 2):
        super().__init__()
        
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        
        # 超网络 - 生成混合网络的权重和偏置
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            agent_q_values: 智能体Q值 [batch_size, actual_num_agents]
            global_state: 全局状态 [batch_size, global_state_dim]
        Returns:
            total_q_value: 全局Q值 [batch_size, 1]
        """
        batch_size, actual_num_agents = agent_q_values.shape
        
        # 如果实际智能体数量与初始化时不同，需要调整
        if actual_num_agents != self.num_agents:
            # 填充或截断到预期的智能体数量
            if actual_num_agents < self.num_agents:
                # 填充零值
                padding = torch.zeros(batch_size, self.num_agents - actual_num_agents, 
                                    device=agent_q_values.device, dtype=agent_q_values.dtype)
                agent_q_values = torch.cat([agent_q_values, padding], dim=1)
            else:
                # 截断到预期数量
                agent_q_values = agent_q_values[:, :self.num_agents]
        
        # 生成第一层权重和偏置
        w1 = torch.abs(self.hyper_w1(global_state))  # 确保权重为正
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state)  # [batch_size, hidden_dim]
        
        # 第一层计算
        agent_q_values = agent_q_values.unsqueeze(-1)  # [batch_size, num_agents, 1]
        hidden = torch.bmm(w1.transpose(1, 2), agent_q_values).squeeze(-1)  # [batch_size, hidden_dim]
        hidden = hidden + b1
        hidden = F.elu(hidden)
        
        # 生成第二层权重和偏置
        w2 = torch.abs(self.hyper_w2(global_state))  # [batch_size, hidden_dim]
        b2 = self.hyper_b2(global_state)  # [batch_size, 1]
        
        # 第二层计算
        total_q_value = torch.sum(w2 * hidden, dim=1, keepdim=True) + b2
        
        return total_q_value


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, *args):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Transition]:
        """采样经验"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class QMIXAgent:
    """QMIX智能体 - 主要的训练和决策类，适配海运脱碳环境"""
    
    def __init__(self, 
                 fleet,  # 船队对象
                 obs_dim: int = 36,  # 海运环境观测维度
                 global_state_dim: int = 82,  # 海运环境全局状态维度
                 lr: float = 5e-4,  # 提高学习率以加快收敛
                 gamma: float = 0.95,  # 降低折扣因子，更关注短期奖励
                 epsilon: float = 0.9,  # 降低初始探索率
                 epsilon_min: float = 0.05,  # 提高最小探索率
                 epsilon_decay: float = 0.998,  # 更慢的探索衰减
                 target_update_freq: int = 100,  # 更频繁的目标网络更新
                 batch_size: int = 64,  # 增大批次大小提高稳定性
                 buffer_size: int = 50000,  # 减小缓冲区大小加快学习
                 hidden_dim: int = 128,  # 减小网络复杂度
                 device: str = 'cpu'):
        
        self.fleet = fleet
        self.num_agents = len(fleet.ships)
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        
        # 构建动作空间映射 - 适配海运脱碳环境
        self.action_dims = {}
        self.action_mappings = {}
        
        # F类船舶动作映射
        f_operating_actions = ['B0', 'B5', 'B7', 'B10', 'B20', 'B30', 'B50', 'B100', 'START_RETROFIT']
        f_retrofitting_actions = ['RETROFIT_CONTINUE']
        
        # G/FG类船舶动作映射
        g_actions = ['G0', 'G5', 'G10', 'G20', 'G50']
        
        # Z-DF类船舶动作映射
        z_actions = ['Z100']
        
        for ship in fleet.ships:
            ship_id = ship.ship_id
            if ship.ship_type.value == 'F':
                if ship.state.value == 'retrofitting':
                    self.action_dims[ship_id] = len(f_retrofitting_actions)
                    self.action_mappings[ship_id] = f_retrofitting_actions
                else:
                    self.action_dims[ship_id] = len(f_operating_actions)
                    self.action_mappings[ship_id] = f_operating_actions
            elif ship.ship_type.value in ['G', 'FG']:
                self.action_dims[ship_id] = len(g_actions)
                self.action_mappings[ship_id] = g_actions
            elif ship.ship_type.value == 'Z_DF':
                self.action_dims[ship_id] = len(z_actions)
                self.action_mappings[ship_id] = z_actions
        
        # 使用最大动作维度作为网络输出维度
        max_action_dim = max(self.action_dims.values())
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # 网络初始化
        self.q_network = QNetwork(obs_dim, max_action_dim, hidden_dim).to(self.device)
        self.target_q_network = QNetwork(obs_dim, max_action_dim, hidden_dim).to(self.device)
        self.mixing_network = MixingNetwork(self.num_agents, global_state_dim, hidden_dim).to(self.device)
        self.target_mixing_network = MixingNetwork(self.num_agents, global_state_dim, hidden_dim).to(self.device)
        
        # 复制参数到目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.mixing_network.parameters()),
            lr=lr
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练计数器
        self.train_step = 0
        
        # 训练历史
        self.training_history = {
            'loss': [],
            'q_values': [],
            'epsilon': [],
            'rewards': []
        }
    
    def get_actions(self, observations: Dict[int, np.ndarray], global_state: np.ndarray, 
                   training: bool = True) -> Dict[int, int]:
        """
        获取动作 - 适配海运脱碳环境的动作空间
        Args:
            observations: 智能体观测 {ship_id: obs}
            global_state: 全局状态
            training: 是否为训练模式
        Returns:
            actions: 智能体动作 {ship_id: action_index}
        """
        actions = {}
        
        # 转换观测为张量
        obs_list = []
        agent_ids = sorted(observations.keys())
        
        for agent_id in agent_ids:
            obs = torch.FloatTensor(observations[agent_id]).unsqueeze(0).to(self.device)
            obs_list.append(obs)
        
        obs_batch = torch.cat(obs_list, dim=0).unsqueeze(0)  # [1, num_agents, obs_dim]
        
        with torch.no_grad():
            q_values = self.q_network(obs_batch)  # [1, num_agents, max_action_dim]
            q_values = q_values.squeeze(0)  # [num_agents, max_action_dim]
        
        # epsilon-greedy策略，考虑每艘船的实际动作空间
        for i, agent_id in enumerate(agent_ids):
            actual_action_dim = self.action_dims[agent_id]
            
            if training and random.random() < self.epsilon:
                # 随机选择动作，限制在实际动作空间内
                action = random.randint(0, actual_action_dim - 1)
            else:
                # 贪婪选择，只考虑实际动作空间
                valid_q_values = q_values[i][:actual_action_dim]
                action = valid_q_values.argmax().item()
            
            actions[agent_id] = action
        
        return actions
    
    def get_action_names(self, actions: Dict[int, int]) -> Dict[int, str]:
        """
        将动作索引转换为动作名称
        Args:
            actions: 动作索引 {ship_id: action_index}
        Returns:
            action_names: 动作名称 {ship_id: action_name}
        """
        action_names = {}
        for ship_id, action_idx in actions.items():
            if ship_id in self.action_mappings:
                action_names[ship_id] = self.action_mappings[ship_id][action_idx]
            else:
                action_names[ship_id] = f"action_{action_idx}"
        return action_names
    
    def store_experience(self, observations: Dict[int, np.ndarray], global_state: np.ndarray,
                        actions: Dict[int, int], rewards: Dict[int, float], 
                        next_observations: Dict[int, np.ndarray], next_global_state: np.ndarray,
                        dones: Dict[int, bool]):
        """存储经验 - 兼容integrated_training.py的接口"""
        self.store_transition(
            state=observations,
            actions=actions,
            rewards=rewards,
            next_state=next_observations,
            dones=dones,
            global_state=global_state,
            next_global_state=next_global_state
        )
    
    def store_transition(self, state: Dict[int, np.ndarray], actions: Dict[int, int], 
                        rewards: Dict[int, float], next_state: Dict[int, np.ndarray], 
                        dones: Dict[int, bool], global_state: np.ndarray, 
                        next_global_state: np.ndarray):
        """存储经验"""
        # 确保所有字典都有相同的键
        common_agent_ids = set(state.keys()) & set(actions.keys()) & set(rewards.keys()) & set(next_state.keys()) & set(dones.keys())
        agent_ids = sorted(common_agent_ids)
        
        # 转换为列表格式
        state_list = [state[agent_id] for agent_id in agent_ids]
        actions_list = [actions[agent_id] for agent_id in agent_ids]
        rewards_list = [rewards[agent_id] for agent_id in agent_ids]
        next_state_list = [next_state[agent_id] for agent_id in agent_ids]
        dones_list = [dones[agent_id] for agent_id in agent_ids]
        
        self.replay_buffer.push(
            np.array(state_list),
            np.array(actions_list),
            np.array(rewards_list),
            np.array(next_state_list),
            np.array(dones_list),
            global_state,
            next_global_state
        )
    
    def train(self) -> Dict[str, float]:
        """训练网络 - 适配海运脱碳环境的动态动作空间"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样经验
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # 处理不同长度的观测数据
        max_agents = max(len(state) for state in batch.state)
        obs_dim = batch.state[0][0].shape[0] if len(batch.state[0]) > 0 else 0
        
        # 创建填充后的批次数据
        batch_size = len(batch.state)
        padded_states = np.zeros((batch_size, max_agents, obs_dim))
        padded_next_states = np.zeros((batch_size, max_agents, obs_dim))
        padded_actions = np.zeros((batch_size, max_agents), dtype=np.int64)
        padded_rewards = np.zeros((batch_size, max_agents))
        padded_dones = np.zeros((batch_size, max_agents), dtype=bool)
        
        for i in range(batch_size):
            num_agents = len(batch.state[i])
            padded_states[i, :num_agents] = batch.state[i]
            padded_next_states[i, :num_agents] = batch.next_state[i]
            padded_actions[i, :num_agents] = batch.actions[i]
            padded_rewards[i, :num_agents] = batch.rewards[i]
            padded_dones[i, :num_agents] = batch.dones[i]
        
        # 转换为张量
        state_batch = torch.FloatTensor(padded_states).to(self.device)
        action_batch = torch.LongTensor(padded_actions).to(self.device)
        reward_batch = torch.FloatTensor(padded_rewards).to(self.device)
        next_state_batch = torch.FloatTensor(padded_next_states).to(self.device)
        done_batch = torch.BoolTensor(padded_dones).to(self.device)
        global_state_batch = torch.FloatTensor(np.array(batch.global_state)).to(self.device)
        next_global_state_batch = torch.FloatTensor(np.array(batch.next_global_state)).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(state_batch)  # [batch_size, num_agents, max_action_dim]
        
        # 处理动态动作空间 - 只选择有效动作的Q值
        batch_size, _, _ = current_q_values.shape
        selected_q_values = torch.zeros(batch_size, max_agents).to(self.device)
        
        for b in range(batch_size):
            actual_agents = len([aid for aid in sorted(self.action_dims.keys())])
            for a in range(min(actual_agents, max_agents)):
                if a < len(sorted(self.action_dims.keys())):
                    agent_id = sorted(self.action_dims.keys())[a]
                    action_idx = action_batch[b, a].item()
                    # 确保动作索引在有效范围内
                    if action_idx < self.action_dims[agent_id]:
                        selected_q_values[b, a] = current_q_values[b, a, action_idx]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state_batch)  # [batch_size, num_agents, max_action_dim]
            
            # 处理动态动作空间 - 只考虑有效动作
            target_q_values = torch.zeros(batch_size, max_agents).to(self.device)
            
            for b in range(batch_size):
                actual_agents = len([aid for aid in sorted(self.action_dims.keys())])
                for a in range(min(actual_agents, max_agents)):
                    if a < len(sorted(self.action_dims.keys())):
                        agent_id = sorted(self.action_dims.keys())[a]
                        actual_action_dim = self.action_dims[agent_id]
                        # 只在有效动作空间内选择最大Q值
                        valid_q_values = next_q_values[b, a, :actual_action_dim]
                        target_q_values[b, a] = valid_q_values.max()
            
            # 计算目标全局Q值
            target_global_q = self.target_mixing_network(target_q_values, next_global_state_batch)
            target_global_q = reward_batch.sum(dim=1, keepdim=True) + self.gamma * target_global_q * (~done_batch.any(dim=1, keepdim=True))
        
        # 计算当前全局Q值
        current_global_q = self.mixing_network(selected_q_values, global_state_batch)
        
        # 计算损失 - 使用Huber损失提高稳定性
        loss = F.smooth_l1_loss(current_global_q, target_global_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 - 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixing_network.parameters()),
            max_norm=5.0  # 降低梯度裁剪阈值
        )
        
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 记录训练历史
        self.training_history['loss'].append(loss.item())
        self.training_history['q_values'].append(current_global_q.mean().item())
        self.training_history['epsilon'].append(self.epsilon)
        
        return {
            'loss': loss.item(),
            'q_value': current_global_q.mean().item(),
            'epsilon': self.epsilon
        }
    
    def save_model(self, filepath: str, ship_strategies=None):
        """保存模型"""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'mixing_network_state_dict': self.mixing_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'target_mixing_network_state_dict': self.target_mixing_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'training_history': self.training_history
        }
        
        # 如果提供了ship_strategies，也保存它
        if ship_strategies is not None:
            save_dict['ship_strategies'] = ship_strategies
            
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.mixing_network.load_state_dict(checkpoint['mixing_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        self.training_history = checkpoint['training_history']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_history['loss']:
            return {}
        
        return {
            'avg_loss': np.mean(self.training_history['loss'][-100:]),
            'avg_q_value': np.mean(self.training_history['q_values'][-100:]),
            'current_epsilon': self.epsilon,
            'train_steps': self.train_step,
            'buffer_size': len(self.replay_buffer)
        }