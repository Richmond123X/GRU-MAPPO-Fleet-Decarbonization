"""基于GRU的MAPPO实现
集成循环神经网络以更好地建模时序依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import copy

from mappo import MAPPOAgent, ActorNetwork, RolloutBuffer
from ship import Ship


class GRUCriticNetwork(nn.Module):
    """基于GRU的Critic网络 - 支持时序状态建模"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影层 - 将观测映射到GRU输入维度
        self.input_projection = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU层 - 核心时序建模组件
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力机制 - 对不同时间步给予不同权重
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 价值头 - 输出最终价值估计
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:  # 只对2维及以上的参数使用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.1)  # 1维参数使用正态分布初始化
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, obs_sequence: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            obs_sequence: 观测序列 [batch_size, seq_len, obs_dim]
            hidden_state: GRU隐藏状态 [num_layers, batch_size, hidden_dim]
        Returns:
            value: 价值估计 [batch_size, 1]
            new_hidden_state: 新的隐藏状态 [num_layers, batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = obs_sequence.shape
        
        # 输入投影
        projected_obs = self.input_projection(obs_sequence.view(-1, self.obs_dim))
        projected_obs = projected_obs.view(batch_size, seq_len, self.hidden_dim)
        
        # GRU处理时序信息
        gru_output, new_hidden_state = self.gru(projected_obs, hidden_state)
        
        # 注意力机制 - 聚合时序信息
        attended_output, attention_weights = self.attention(
            gru_output, gru_output, gru_output
        )
        
        # 取最后一个时间步的输出作为当前状态的价值估计
        final_output = attended_output[:, -1, :]  # [batch_size, hidden_dim]
        
        # 价值估计
        value = self.value_head(final_output)  # [batch_size, 1]
        
        return value, new_hidden_state


class SequenceRolloutBuffer:
    """支持序列数据的经验回放缓冲区"""
    
    def __init__(self, max_size: int, obs_dim: int, action_dims: Dict[int, int], 
                 sequence_length: int = 10, device: str = 'cpu'):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.sequence_length = sequence_length
        self.device = device
        
        # 存储结构
        self.observations = {}  # {agent_id: deque of observations}
        self.actions = {}       # {agent_id: deque of actions}
        self.rewards = {}       # {agent_id: deque of rewards}
        self.values = {}        # {agent_id: deque of values}
        self.log_probs = {}     # {agent_id: deque of log_probs}
        self.dones = {}         # {agent_id: deque of dones}
        self.hidden_states = {} # {agent_id: deque of hidden_states}
        
        # 初始化每个智能体的缓冲区
        for agent_id in action_dims.keys():
            self.observations[agent_id] = deque(maxlen=max_size)
            self.actions[agent_id] = deque(maxlen=max_size)
            self.rewards[agent_id] = deque(maxlen=max_size)
            self.values[agent_id] = deque(maxlen=max_size)
            self.log_probs[agent_id] = deque(maxlen=max_size)
            self.dones[agent_id] = deque(maxlen=max_size)
            self.hidden_states[agent_id] = deque(maxlen=max_size)
        
        self.size = 0
    
    def add(self, observations: Dict[int, np.ndarray], actions: Dict[int, np.ndarray],
            rewards: Dict[int, float], values: Dict[int, float], 
            log_probs: Dict[int, float], dones: Dict[int, bool],
            hidden_states: Dict[int, torch.Tensor]):
        """添加一步经验"""
        for agent_id, obs in observations.items():
            if agent_id in self.action_dims:
                self.observations[agent_id].append(obs)
                self.actions[agent_id].append(actions.get(agent_id, 0))
                self.rewards[agent_id].append(rewards.get(agent_id, 0.0))
                self.values[agent_id].append(values.get(agent_id, 0.0))
                self.log_probs[agent_id].append(log_probs.get(agent_id, 0.0))
                self.dones[agent_id].append(dones.get(agent_id, False))
                self.hidden_states[agent_id].append(hidden_states.get(agent_id, torch.zeros(1, 1, 256)))
        
        self.size = min(self.size + 1, self.max_size)
    
    def store(self, observations: Dict[int, np.ndarray], 
              global_obs: np.ndarray,
              actions: Dict[int, int], 
              log_probs: Dict[int, float],
              rewards: Dict[int, float], 
              dones: Dict[int, bool],
              values: Dict[int, float]):
        """存储经验（与add方法兼容的接口）"""
        # 创建虚拟隐藏状态
        hidden_states = {}
        for agent_id in self.action_dims.keys():
            hidden_states[agent_id] = torch.zeros(1, 1, 256)  # 虚拟隐藏状态
        
        self.add(observations, actions, rewards, values, log_probs, dones, hidden_states)
    
    def get_sequences(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """获取指定智能体的序列数据"""
        if len(self.observations[agent_id]) < self.sequence_length:
            return None
        
        # 构建序列
        sequences = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'hidden_states': []
        }
        
        obs_list = list(self.observations[agent_id])
        actions_list = list(self.actions[agent_id])
        rewards_list = list(self.rewards[agent_id])
        values_list = list(self.values[agent_id])
        log_probs_list = list(self.log_probs[agent_id])
        dones_list = list(self.dones[agent_id])
        hidden_states_list = list(self.hidden_states[agent_id])
        
        # 创建滑动窗口序列
        for i in range(len(obs_list) - self.sequence_length + 1):
            sequences['observations'].append(obs_list[i:i+self.sequence_length])
            sequences['actions'].append(actions_list[i:i+self.sequence_length])
            sequences['rewards'].append(rewards_list[i:i+self.sequence_length])
            sequences['values'].append(values_list[i:i+self.sequence_length])
            sequences['log_probs'].append(log_probs_list[i:i+self.sequence_length])
            sequences['dones'].append(dones_list[i:i+self.sequence_length])
            sequences['hidden_states'].append(hidden_states_list[i:i+self.sequence_length])
        
        # 转换为张量
        for key in sequences:
            if key == 'hidden_states':
                # 隐藏状态需要特殊处理
                sequences[key] = torch.stack([torch.stack(seq) for seq in sequences[key]])
            else:
                import numpy as np
                arr = np.array(sequences[key], dtype=np.float32)
                sequences[key] = torch.from_numpy(arr).to(self.device)
        
        return sequences
    
    def reset(self):
        """重置缓冲区"""
        for agent_id in self.action_dims.keys():
            self.observations[agent_id].clear()
            self.actions[agent_id].clear()
            self.rewards[agent_id].clear()
            self.values[agent_id].clear()
            self.log_probs[agent_id].clear()
            self.dones[agent_id].clear()
            self.hidden_states[agent_id].clear()
        self.size = 0
    
    def clear(self):
        """清空缓冲区（与reset相同）"""
        self.reset()
    
    def is_ready(self):
        """检查缓冲区是否准备好进行训练"""
        # 检查是否有足够的数据进行训练
        if self.size == 0:
            return False
        
        # 检查每个智能体是否都有数据
        for agent_id in self.action_dims.keys():
            if len(self.observations[agent_id]) == 0:
                return False
        
        return True
    
    def compute_gae_returns(self, next_value: float, gamma: float, gae_lambda: float):
        """计算GAE优势和回报"""
        # 简化实现：直接使用奖励作为优势
        # 在实际应用中，这里应该实现完整的GAE计算
        pass


class GRUMAPPOAgent:
    """基于GRU的MAPPO智能体"""
    
    def __init__(self, 
                 obs_dim: int,
                 global_obs_dim: int,
                 action_dims: Dict[int, int],
                 max_episode_length: int = 50,
                 sequence_length: int = 10,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 device: str = 'cpu',
                 hidden_dim: int = 256):
        
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dims = action_dims
        self.max_episode_length = max_episode_length
        self.sequence_length = sequence_length
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 超参数
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # 创建网络
        self.actors = {}
        for agent_id, action_dim in action_dims.items():
            self.actors[agent_id] = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        
        # GRU Critic网络
        self.critic = GRUCriticNetwork(obs_dim, hidden_dim).to(device)
        
        # 优化器
        actor_params = []
        for actor in self.actors.values():
            actor_params.extend(actor.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 序列缓冲区
        self.rollout_buffer = SequenceRolloutBuffer(
            max_size=max_episode_length * 10,
            obs_dim=obs_dim,
            action_dims=action_dims,
            sequence_length=sequence_length,
            device=device
        )
        
        # GRU隐藏状态管理
        self.hidden_states = {}
        self.reset_hidden_states()
        
        # 训练统计
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'clip_fractions': [],
            'kl_divergences': []
        }
    
    def reset_hidden_states(self):
        """重置所有智能体的GRU隐藏状态"""
        for agent_id in self.action_dims.keys():
            self.hidden_states[agent_id] = torch.zeros(
                self.critic.num_layers, 1, self.hidden_dim
            ).to(self.device)
    
    def get_actions(self, observations: Dict[int, np.ndarray], 
                   deterministic: bool = False) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        """获取动作和价值估计"""
        actions = {}
        log_probs = {}
        values = {}
        
        with torch.no_grad():
            for agent_id, obs in observations.items():
                if agent_id in self.actors:
                    # Actor网络获取动作
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_logits = self.actors[agent_id](obs_tensor)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    
                    if deterministic:
                        action = action_dist.mode
                    else:
                        action = action_dist.sample()
                    
                    log_prob = action_dist.log_prob(action)
                    
                    actions[agent_id] = action.item()  # 转换为标量整数
                    log_probs[agent_id] = log_prob.item()
                    
                    # GRU Critic网络获取价值估计
                    obs_sequence = obs_tensor.unsqueeze(1)  # [1, 1, obs_dim]
                    value, new_hidden = self.critic(obs_sequence, self.hidden_states[agent_id])
                    values[agent_id] = value.item()
                    
                    # 更新隐藏状态
                    self.hidden_states[agent_id] = new_hidden
        
        return actions, log_probs, values
    
    def store_transition(self, observations: Dict[int, np.ndarray], 
                        global_obs: np.ndarray,
                        actions: Dict[int, int], 
                        log_probs: Dict[int, float],
                        rewards: Dict[int, float], 
                        dones: Dict[int, bool],
                        values: Dict[int, float]):
        """存储经验到序列缓冲区"""
        self.rollout_buffer.store(
            observations=observations,
            global_obs=global_obs,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values
        )
    
    def get_value(self, global_obs: np.ndarray) -> float:
        """获取全局状态价值估计"""
        with torch.no_grad():
            # 使用第一个智能体的critic网络估计全局价值
            obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).unsqueeze(1).to(self.device)
            first_agent_id = list(self.action_dims.keys())[0]
            value, _ = self.critic(obs_tensor, self.hidden_states[first_agent_id])
            return value.item()
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """更新网络参数"""
        # 计算GAE优势和回报
        self.rollout_buffer.compute_gae_returns(next_value, self.gamma, self.gae_lambda)
        
        # 训练网络
        update_stats = self.train(self.rollout_buffer)
        
        # 清空缓冲区
        self.rollout_buffer.clear()
        
        # 重置隐藏状态
        self.reset_hidden_states()
        
        return update_stats
    
    def train(self, rollout_buffer: SequenceRolloutBuffer) -> Dict[str, float]:
        """训练GRU-MAPPO智能体"""
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        total_kl_divergence = 0
        num_updates = 0
        
        for epoch in range(self.ppo_epochs):
            for agent_id in self.action_dims.keys():
                sequences = rollout_buffer.get_sequences(agent_id)
                if sequences is None:
                    continue
                
                # 计算优势和回报
                advantages, returns = self._compute_gae(sequences)
                
                # 训练Actor
                actor_loss, entropy, clip_fraction, kl_div = self._train_actor(
                    agent_id, sequences, advantages
                )
                
                # 训练Critic
                critic_loss = self._train_critic(sequences, returns)
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy
                total_clip_fraction += clip_fraction
                total_kl_divergence += kl_div
                num_updates += 1
        
        # 计算平均损失
        if num_updates > 0:
            avg_stats = {
                'actor_loss': total_actor_loss / num_updates,
                'critic_loss': total_critic_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'clip_fraction': total_clip_fraction / num_updates,
                'kl_divergence': total_kl_divergence / num_updates
            }
        else:
            avg_stats = {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'clip_fraction': 0.0,
                'kl_divergence': 0.0
            }
        
        # 更新训练统计
        for key, value in avg_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return avg_stats
    
    def _compute_gae(self, sequences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计"""
        rewards = sequences['rewards']
        values = sequences['values']
        dones = sequences['dones']
        
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        for b in range(batch_size):
            gae = 0
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0
                else:
                    next_value = values[b, t + 1]
                
                delta = rewards[b, t] + self.gamma * next_value * (1 - dones[b, t]) - values[b, t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[b, t]) * gae
                advantages[b, t] = gae
                returns[b, t] = advantages[b, t] + values[b, t]
        
        return advantages, returns
    
    def _train_actor(self, agent_id: int, sequences: Dict[str, torch.Tensor], 
                    advantages: torch.Tensor) -> Tuple[float, float, float, float]:
        """训练Actor网络"""
        observations = sequences['observations']
        actions = sequences['actions']
        old_log_probs = sequences['log_probs']
        
        batch_size, seq_len, obs_dim = observations.shape
        
        # 重塑数据
        obs_flat = observations.view(-1, obs_dim)
        actions_flat = actions.view(-1).long()
        old_log_probs_flat = old_log_probs.view(-1)
        advantages_flat = advantages.view(-1)
        
        # 标准化优势
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # 前向传播
        action_logits = self.actors[agent_id](obs_flat)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        new_log_probs = action_dist.log_prob(actions_flat)
        entropy = action_dist.entropy().mean()
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs_flat)
        
        # PPO损失
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_flat
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 总损失
        total_loss = actor_loss - self.entropy_coef * entropy
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # 统计信息
        with torch.no_grad():
            clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean()
            kl_divergence = (old_log_probs_flat - new_log_probs).mean()
        
        return actor_loss.item(), entropy.item(), clip_fraction.item(), kl_divergence.item()
    
    def _train_critic(self, sequences: Dict[str, torch.Tensor], returns: torch.Tensor) -> float:
        """训练GRU Critic网络"""
        observations = sequences['observations']
        
        # 前向传播
        values, _ = self.critic(observations)
        values = values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        
        # returns是[batch_size, seq_len]，但我们只需要最后一个时间步的值
        if returns.dim() == 2:
            returns = returns[:, -1]  # 取最后一个时间步 [batch_size]
        
        # 价值损失
        critic_loss = F.mse_loss(values, returns)
        
        # 反向传播
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def save_model(self, path: str, ship_strategies=None):
        """保存模型"""
        checkpoint = {
            'actors': {str(agent_id): actor.state_dict() for agent_id, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        
        # 如果提供了ship_strategies，也保存它
        if ship_strategies is not None:
            checkpoint['ship_strategies'] = ship_strategies
            
        torch.save(checkpoint, path)
        print(f"GRU-MAPPO模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载网络参数
        for agent_id, state_dict in checkpoint['actors'].items():
            if int(agent_id) in self.actors:
                self.actors[int(agent_id)].load_state_dict(state_dict)
        
        self.critic.load_state_dict(checkpoint['critic'])
        
        # 加载优化器参数
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # 加载训练统计
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        print(f"GRU-MAPPO模型已从 {path} 加载")
