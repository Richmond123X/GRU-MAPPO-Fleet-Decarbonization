"""标准MAPPO算法实现
多智能体近端策略优化算法 (Multi-Agent Proximal Policy Optimization)
采用集中式训练分散式执行的架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy

class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 增强的特征提取层 - 更深更宽的网络提高GPU利用率
        self.feature_extractor = nn.Sequential(
            # 第一层：扩展维度
            nn.Linear(obs_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二层：保持高维度
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三层：继续高维度计算
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第四层：额外的深度
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第五层：逐渐收缩
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第六层：最终特征层
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # 策略头使用较小的初始化
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_extractor(obs)
        logits = self.policy_head(features)
        return logits
    
    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布"""
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)
    
    def get_action_log_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """获取动作的对数概率"""
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """计算策略熵"""
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络
    在MAPPO中，Critic网络接收全局状态信息进行集中式价值估计
    """
    
    def __init__(self, global_obs_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.global_obs_dim = global_obs_dim
        
        # 全局状态编码器（使用LayerNorm替代BatchNorm，避免批次大小为1的问题）
        self.global_encoder = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout提高泛化能力
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 价值头
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # 价值头使用单位初始化
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.global_encoder(global_obs)
        value = self.value_head(features)
        return value.squeeze(-1)

class RolloutBuffer:
    """轨迹缓冲区 - 存储一个episode的完整轨迹"""
    
    def __init__(self, max_episode_length: int, num_agents: int, obs_dim: int, global_obs_dim: int):
        self.max_episode_length = max_episode_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        
        # 初始化缓冲区
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.observations = []
        self.global_observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.step_count = 0
    
    def store(self, obs: Dict[int, np.ndarray], global_obs: np.ndarray, 
              actions: Dict[int, int], log_probs: Dict[int, float],
              rewards: Dict[int, float], values: float, dones: Dict[int, bool]):
        """存储一步的经验"""
        self.observations.append(obs.copy())
        self.global_observations.append(global_obs.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.rewards.append(rewards.copy())
        self.values.append(values)
        self.dones.append(dones.copy())
        self.step_count += 1
    
    def get_data(self) -> Dict[str, Any]:
        """获取缓冲区数据"""
        return {
            'observations': self.observations,
            'global_observations': self.global_observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'rewards': self.rewards,
            'values': self.values,
            'dones': self.dones,
            'step_count': self.step_count
        }
    
    def __len__(self):
        return self.step_count

class MAPPOAgent:
    """MAPPO智能体
    实现多智能体近端策略优化算法
    采用集中式训练分散式执行的架构
    """
    
    def __init__(self, 
                 obs_dim: int,
                 global_obs_dim: int,
                 action_dims: Dict[int, int],
                 max_episode_length: int = 50,
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
                 device: str = 'cpu'):
        
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dims = action_dims
        self.max_episode_length = max_episode_length
        self.lr_actor = lr_actor  # 保存学习率为实例属性
        self.lr_critic = lr_critic  # 保存学习率为实例属性
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(device)
        
        # 初始化混合精度训练和梯度累积
        if device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.gradient_accumulation_steps = 4  # 梯度累积步数
        else:
            self.scaler = None
            self.gradient_accumulation_steps = 1
            
            # CPU优化：使用合理的线程数，专注于向量化计算
            import os
            if hasattr(torch, 'set_num_threads'):
                try:
                    num_threads = min(os.cpu_count(), 4)  # 进一步减少线程数
                    # 检查是否已经设置过线程数
                    current_threads = torch.get_num_threads()
                    if current_threads != num_threads:
                        torch.set_num_threads(num_threads)
                    print(f"  ✓ MAPPO智能体CPU线程优化: {torch.get_num_threads()}线程 (向量化计算)")
                except RuntimeError:
                    # 如果已经设置过，忽略错误
                    print(f"  ✓ MAPPO智能体使用已设置的CPU线程配置: {torch.get_num_threads()}线程")
        
        # 为每个智能体创建Actor网络
        self.actors = {}
        self.actor_optimizers = {}
        
        for agent_id, action_dim in action_dims.items():
            actor = ActorNetwork(obs_dim, action_dim).to(self.device)
            self.actors[agent_id] = actor
            self.actor_optimizers[agent_id] = optim.Adam(actor.parameters(), lr=lr_actor)
        
        # 共享Critic网络
        self.critic = CriticNetwork(global_obs_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 学习率调度器 - 支持多种调度策略
        from config import training_config
        
        self.actor_schedulers = {}
        for agent_id in action_dims.keys():
            self.actor_schedulers[agent_id] = self._create_scheduler(
                self.actor_optimizers[agent_id], lr_actor, training_config.LR_SCHEDULER_TYPE
            )
        
        self.critic_scheduler = self._create_scheduler(
            self.critic_optimizer, lr_critic, training_config.LR_SCHEDULER_TYPE
        )
        
        # 早停机制相关变量
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.early_stopping_enabled = training_config.EARLY_STOPPING
        
        # 轨迹缓冲区
        self.buffer = RolloutBuffer(
            max_episode_length=max_episode_length,
            num_agents=len(action_dims),
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim
        )
        
        # 训练统计
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'clip_fractions': [],
            'explained_variances': [],
            'kl_divergences': []
        }
    
    def get_actions(self, observations: Dict[int, np.ndarray], 
                   deterministic: bool = False) -> Tuple[Dict[int, int], Dict[int, float]]:
        """获取动作"""
        actions = {}
        log_probs = {}
        
        with torch.no_grad():
            for agent_id, obs in observations.items():
                if agent_id not in self.actors:
                    continue
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                if deterministic:
                    # 确定性动作（测试时使用）
                    logits = self.actors[agent_id](obs_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                    log_prob = 0.0
                else:
                    # 随机动作（训练时使用）
                    logits = self.actors[agent_id](obs_tensor)
                    probs = F.softmax(logits, dim=-1)
                    action_dist = torch.distributions.Categorical(probs)
                    action = action_dist.sample().item()
                    log_prob = action_dist.log_prob(torch.tensor(action, device=self.device)).item()
                
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
        
        return actions, log_probs
    
    def get_value(self, global_obs: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
            value = self.critic(global_obs_tensor).item()
        return value
    
    def store_transition(self, 
                        observations: Dict[int, np.ndarray],
                        global_obs: np.ndarray,
                        actions: Dict[int, int],
                        log_probs: Dict[int, float],
                        rewards: Dict[int, float],
                        dones: Dict[int, bool]):
        """存储转换"""
        # 计算当前状态价值
        value = self.get_value(global_obs)
        
        # 存储到缓冲区
        self.buffer.store(
            obs=observations,
            global_obs=global_obs,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=value,
            dones=dones
        )
    
    def compute_gae_advantages(self, buffer_data: Dict[str, Any], 
                              next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势估计"""
        rewards_list = buffer_data['rewards']
        values_list = buffer_data['values']
        dones_list = buffer_data['dones']
        
        # 为每个智能体计算优势
        agent_advantages = {agent_id: [] for agent_id in self.actors.keys()}
        agent_returns = {agent_id: [] for agent_id in self.actors.keys()}
        
        for agent_id in self.actors.keys():
            # 收集该智能体的奖励和完成状态
            agent_rewards = []
            agent_dones = []
            
            for step_rewards, step_dones in zip(rewards_list, dones_list):
                agent_rewards.append(step_rewards.get(agent_id, 0.0))
                agent_dones.append(step_dones.get(agent_id, False))
            
            # 计算GAE
            advantages = []
            returns = []
            gae = 0
            
            # 从后往前计算
            for i in reversed(range(len(agent_rewards))):
                if i == len(agent_rewards) - 1:
                    next_non_terminal = 1.0 - agent_dones[i]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - agent_dones[i]
                    next_val = values_list[i + 1]
                
                delta = agent_rewards[i] + self.gamma * next_val * next_non_terminal - values_list[i]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                
                advantages.insert(0, gae)
                returns.insert(0, gae + values_list[i])
            
            agent_advantages[agent_id] = advantages
            agent_returns[agent_id] = returns
        
        return agent_advantages, agent_returns
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.buffer) == 0:
            return {}
        
        # 获取缓冲区数据
        buffer_data = self.buffer.get_data()
        
        # 计算优势和回报
        advantages, returns = self.compute_gae_advantages(buffer_data, next_value)
        
        # 准备训练数据
        train_data = self._prepare_training_data(buffer_data, advantages, returns)
        
        if not train_data:
            return {}
        
        # 多轮PPO更新
        total_stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            'kl_divergence': 0.0
        }
        
        for epoch in range(self.ppo_epochs):
            epoch_stats = self._ppo_update(train_data)
            for key, value in epoch_stats.items():
                total_stats[key] += value
        
        # 平均统计
        for key in total_stats:
            total_stats[key] /= self.ppo_epochs
        
        # 记录统计信息
        for key, value in total_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        # 更新学习率调度器
        self.update_lr_scheduler()
        
        # 清空缓冲区
        self.buffer.reset()
        
        return total_stats
    
    def _prepare_training_data(self, buffer_data: Dict[str, Any], 
                              advantages: Dict[int, List[float]], 
                              returns: Dict[int, List[float]]) -> Dict[str, torch.Tensor]:
        """准备训练数据 - 向量化优化版本"""
        observations_list = buffer_data['observations']
        global_observations_list = buffer_data['global_observations']
        actions_list = buffer_data['actions']
        log_probs_list = buffer_data['log_probs']
        values_list = buffer_data['values']
        
        # 预分配数组以减少内存重分配
        num_steps = len(observations_list)
        num_agents = len(self.actors)
        total_samples = num_steps * num_agents
        
        # 使用NumPy数组预分配内存
        obs_shape = observations_list[0][list(self.actors.keys())[0]].shape
        global_obs_shape = global_observations_list[0].shape
        
        all_obs = np.empty((total_samples, *obs_shape), dtype=np.float32)
        all_global_obs = np.empty((total_samples, *global_obs_shape), dtype=np.float32)
        all_actions = np.empty(total_samples, dtype=np.int64)
        all_old_log_probs = np.empty(total_samples, dtype=np.float32)
        all_advantages = np.empty(total_samples, dtype=np.float32)
        all_returns = np.empty(total_samples, dtype=np.float32)
        all_values = np.empty(total_samples, dtype=np.float32)
        all_agent_ids = []
        
        # 向量化数据收集
        idx = 0
        agent_ids = list(self.actors.keys())
        
        for step in range(num_steps):
            for agent_id in agent_ids:
                if (agent_id in observations_list[step] and 
                    agent_id in advantages and 
                    step < len(advantages[agent_id])):
                    
                    all_obs[idx] = observations_list[step][agent_id]
                    all_global_obs[idx] = global_observations_list[step]
                    all_actions[idx] = actions_list[step][agent_id]
                    all_old_log_probs[idx] = log_probs_list[step][agent_id]
                    all_advantages[idx] = advantages[agent_id][step]
                    all_returns[idx] = returns[agent_id][step]
                    all_values[idx] = values_list[step]
                    all_agent_ids.append(agent_id)
                    idx += 1
        
        if idx == 0:
            return {}
        
        # 截取实际使用的数据
        all_obs = all_obs[:idx]
        all_global_obs = all_global_obs[:idx]
        all_actions = all_actions[:idx]
        all_old_log_probs = all_old_log_probs[:idx]
        all_advantages = all_advantages[:idx]
        all_returns = all_returns[:idx]
        all_values = all_values[:idx]
        
        # 直接从NumPy创建张量，避免额外拷贝
        train_data = {
            'observations': torch.from_numpy(all_obs).to(self.device, non_blocking=True),
            'global_observations': torch.from_numpy(all_global_obs).to(self.device, non_blocking=True),
            'actions': torch.from_numpy(all_actions).to(self.device, non_blocking=True),
            'old_log_probs': torch.from_numpy(all_old_log_probs).to(self.device, non_blocking=True),
            'advantages': torch.from_numpy(all_advantages).to(self.device, non_blocking=True),
            'returns': torch.from_numpy(all_returns).to(self.device, non_blocking=True),
            'old_values': torch.from_numpy(all_values).to(self.device, non_blocking=True),
            'agent_ids': all_agent_ids
        }
        
        # 向量化标准化优势
        if len(train_data['advantages']) > 1:
            adv_mean = train_data['advantages'].mean()
            adv_std = train_data['advantages'].std()
            train_data['advantages'] = (train_data['advantages'] - adv_mean) / (adv_std + 1e-8)
        
        return train_data
    
    def _ppo_update(self, train_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO更新 - 支持梯度累积"""
        # 将数据分成小批次进行梯度累积
        total_samples = len(train_data['observations'])
        mini_batch_size = max(1, total_samples // self.gradient_accumulation_steps)
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_kl_div = 0.0
        
        # 清零梯度
        self.critic_optimizer.zero_grad()
        for optimizer in self.actor_optimizers.values():
            optimizer.zero_grad()
        
        # 分批次进行梯度累积
        for step in range(self.gradient_accumulation_steps):
            start_idx = step * mini_batch_size
            end_idx = min((step + 1) * mini_batch_size, total_samples)
            
            if start_idx >= total_samples:
                break
                
            # 创建小批次数据
            mini_batch = {}
            for key, value in train_data.items():
                if isinstance(value, torch.Tensor):
                    mini_batch[key] = value[start_idx:end_idx]
                else:
                    mini_batch[key] = value[start_idx:end_idx]
            
            # 更新Critic (累积梯度)
            critic_loss = self._update_critic_accumulate(mini_batch, step == self.gradient_accumulation_steps - 1)
            total_critic_loss += critic_loss
            
            # 更新Actors (累积梯度)
            actor_loss, entropy, clip_fraction, kl_div = self._update_actors_accumulate(
                mini_batch, step == self.gradient_accumulation_steps - 1)
            total_actor_loss += actor_loss
            total_entropy += entropy
            total_clip_fraction += clip_fraction
            total_kl_div += kl_div
            
            # 清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 计算解释方差
        explained_var = self._compute_explained_variance(
            train_data['returns'], 
            train_data['global_observations']
        )
        
        return {
            'actor_loss': total_actor_loss / self.gradient_accumulation_steps,
            'critic_loss': total_critic_loss / self.gradient_accumulation_steps,
            'entropy': total_entropy / self.gradient_accumulation_steps,
            'clip_fraction': total_clip_fraction / self.gradient_accumulation_steps,
            'explained_variance': explained_var,
            'kl_divergence': total_kl_div / self.gradient_accumulation_steps
        }
    
    def _update_critic_accumulate(self, train_data: Dict[str, torch.Tensor], is_last_step: bool) -> float:
        """更新Critic网络 - 支持梯度累积"""
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            # 计算价值损失
            values = self.critic(train_data['global_observations'])
            
            # 可选：使用价值函数裁剪
            values_clipped = train_data['old_values'] + torch.clamp(
                values - train_data['old_values'], -self.clip_ratio, self.clip_ratio
            )
            
            value_loss1 = F.mse_loss(values, train_data['returns'])
            value_loss2 = F.mse_loss(values_clipped, train_data['returns'])
            value_loss = torch.max(value_loss1, value_loss2)
            
            # 梯度累积：除以累积步数
            scaled_loss = (self.value_coef * value_loss) / self.gradient_accumulation_steps
        
        # 反向传播 - 累积梯度
        if hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            
            # 只在最后一步更新参数
            if is_last_step:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
        else:
            scaled_loss.backward()
            
            # 只在最后一步更新参数
            if is_last_step:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        
        return value_loss.item()

    def _update_actors_accumulate(self, train_data: Dict[str, torch.Tensor], is_last_step: bool) -> Tuple[float, float, float, float]:
        """更新Actor网络 - 支持梯度累积"""
        total_actor_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_kl_div = 0.0
        num_updates = 0
        
        # 按智能体分组更新
        agent_indices = {}
        for i, agent_id in enumerate(train_data['agent_ids']):
            if agent_id not in agent_indices:
                agent_indices[agent_id] = []
            agent_indices[agent_id].append(i)
        
        for agent_id, indices in agent_indices.items():
            if agent_id not in self.actors:
                continue
                
            # 提取该智能体的数据
            agent_obs = train_data['observations'][indices]
            agent_actions = train_data['actions'][indices]
            agent_old_log_probs = train_data['old_log_probs'][indices]
            agent_advantages = train_data['advantages'][indices]
            
            with torch.cuda.amp.autocast():
                # 计算新的动作概率
                logits = self.actors[agent_id](agent_obs)
                probs = F.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                
                # 计算新的log概率
                new_log_probs = action_dist.log_prob(agent_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - agent_old_log_probs)
                
                # 计算PPO损失
                surr1 = ratio * agent_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * agent_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算熵
                entropy = action_dist.entropy().mean()
                
                # 总损失（包含熵正则化）
                total_loss = actor_loss - self.entropy_coef * entropy
                
                # 梯度累积：除以累积步数
                scaled_loss = total_loss / self.gradient_accumulation_steps
            
            # 反向传播 - 累积梯度
            if hasattr(self, 'scaler') and self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
                
                # 只在最后一步更新参数
                if is_last_step:
                    self.scaler.unscale_(self.actor_optimizers[agent_id])
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                    self.scaler.step(self.actor_optimizers[agent_id])
            else:
                scaled_loss.backward()
                
                # 只在最后一步更新参数
                if is_last_step:
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                    self.actor_optimizers[agent_id].step()
            
            # 统计信息
            total_actor_loss += actor_loss.item()
            total_entropy += entropy.item()
            
            # 计算裁剪比例
            with torch.no_grad():
                clip_fraction = ((ratio < 1 - self.clip_ratio) | (ratio > 1 + self.clip_ratio)).float().mean()
                total_clip_fraction += clip_fraction.item()
                
                # 计算KL散度
                kl_div = (agent_old_log_probs - new_log_probs).mean()
                total_kl_div += kl_div.item()
            
            num_updates += 1
        
        # 只在最后一步更新scaler
        if is_last_step and hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler.update()
        
        if num_updates == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        return (total_actor_loss / num_updates, 
                total_entropy / num_updates,
                total_clip_fraction / num_updates,
                total_kl_div / num_updates)
    
    def _update_critic(self, train_data: Dict[str, torch.Tensor]) -> float:
        """更新Critic网络 - 支持混合精度训练"""
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            # 计算价值损失
            values = self.critic(train_data['global_observations'])
            
            # 可选：使用价值函数裁剪
            values_clipped = train_data['old_values'] + torch.clamp(
                values - train_data['old_values'], -self.clip_ratio, self.clip_ratio
            )
            
            value_loss1 = F.mse_loss(values, train_data['returns'])
            value_loss2 = F.mse_loss(values_clipped, train_data['returns'])
            value_loss = torch.max(value_loss1, value_loss2)
        
        # 反向传播 - 使用梯度缩放
        self.critic_optimizer.zero_grad()
        if hasattr(self, 'scaler'):
            self.scaler.scale(self.value_coef * value_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            (self.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        return value_loss.item()
    
    def _update_actors(self, train_data: Dict[str, torch.Tensor]) -> Tuple[float, float, float, float]:
        """更新Actor网络"""
        total_actor_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_kl_div = 0.0
        num_updates = 0
        
        # 按智能体分组更新
        agent_indices = {}
        for i, agent_id in enumerate(train_data['agent_ids']):
            if agent_id not in agent_indices:
                agent_indices[agent_id] = []
            agent_indices[agent_id].append(i)
        
        for agent_id, indices in agent_indices.items():
            if agent_id not in self.actors:
                continue
            
            # 提取该智能体的数据
            agent_obs = train_data['observations'][indices]
            agent_actions = train_data['actions'][indices]
            agent_old_log_probs = train_data['old_log_probs'][indices]
            agent_advantages = train_data['advantages'][indices]
            
            # 计算新的对数概率和熵
            new_log_probs = self.actors[agent_id].get_action_log_probs(agent_obs, agent_actions)
            entropy = self.actors[agent_id].get_entropy(agent_obs).mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - agent_old_log_probs)
            
            # PPO损失
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * agent_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 总损失
            actor_loss = policy_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()
            
            # 统计
            total_actor_loss += actor_loss.item()
            total_entropy += entropy.item()
            
            # 计算裁剪比例和KL散度
            with torch.no_grad():
                clip_mask = (ratio < 1 - self.clip_ratio) | (ratio > 1 + self.clip_ratio)
                total_clip_fraction += clip_mask.float().mean().item()
                
                # 近似KL散度
                kl_div = (agent_old_log_probs - new_log_probs).mean().item()
                total_kl_div += abs(kl_div)
            
            num_updates += 1
        
        if num_updates > 0:
            total_actor_loss /= num_updates
            total_entropy /= num_updates
            total_clip_fraction /= num_updates
            total_kl_div /= num_updates
        
        return total_actor_loss, total_entropy, total_clip_fraction, total_kl_div
    
    def _compute_explained_variance(self, returns: torch.Tensor, global_obs: torch.Tensor) -> float:
        """计算解释方差"""
        with torch.no_grad():
            values = self.critic(global_obs)
            var_returns = torch.var(returns)
            var_residual = torch.var(returns - values)
            
            if var_returns > 0:
                explained_var = 1 - var_residual / var_returns
            else:
                explained_var = 0.0
            
            return explained_var.item()
    
    def save_models(self, filepath: str, ship_strategies=None):
        """保存模型"""
        checkpoint = {
            'actors': {agent_id: actor.state_dict() for agent_id, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
            'actor_optimizers': {agent_id: opt.state_dict() for agent_id, opt in self.actor_optimizers.items()},
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_schedulers': {agent_id: sch.state_dict() for agent_id, sch in self.actor_schedulers.items()},
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'training_stats': self.training_stats
        }
        
        # 如果提供了船舶策略，也保存到检查点中
        if ship_strategies is not None:
            checkpoint['ship_strategies'] = ship_strategies
            
        torch.save(checkpoint, filepath)
    
    def load_models(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for agent_id, state_dict in checkpoint['actors'].items():
            if agent_id in self.actors:
                self.actors[agent_id].load_state_dict(state_dict)
        
        self.critic.load_state_dict(checkpoint['critic'])
        
        for agent_id, state_dict in checkpoint['actor_optimizers'].items():
            if agent_id in self.actor_optimizers:
                self.actor_optimizers[agent_id].load_state_dict(state_dict)
        
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # 加载学习率调度器参数
        if 'actor_schedulers' in checkpoint:
            for agent_id, state_dict in checkpoint['actor_schedulers'].items():
                if agent_id in self.actor_schedulers:
                    self.actor_schedulers[agent_id].load_state_dict(state_dict)
        
        if 'critic_scheduler' in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return self.training_stats.copy()
    
    def _create_scheduler(self, optimizer, initial_lr: float, scheduler_type: str):
        """创建学习率调度器"""
        from config import training_config
        
        if scheduler_type == "none":
            # 返回一个不做任何操作的调度器
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=training_config.COSINE_T_MAX,
                eta_min=initial_lr * training_config.COSINE_ETA_MIN_RATIO
            )
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.LR_DECAY_STEP,
                gamma=training_config.LR_DECAY_GAMMA
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.STEP_SIZE,
                gamma=training_config.STEP_GAMMA
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # 监控奖励最大化
                factor=training_config.PLATEAU_FACTOR,
                patience=training_config.PLATEAU_PATIENCE,
                threshold=training_config.PLATEAU_THRESHOLD,
                verbose=True
            )
        else:
            # 默认使用改进的余弦退火
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=training_config.COSINE_T_MAX,
                eta_min=initial_lr * training_config.COSINE_ETA_MIN_RATIO
            )
    
    def check_early_stopping(self, current_reward: float) -> bool:
        """检查是否应该早停"""
        from config import training_config
        
        if not self.early_stopping_enabled:
            return False
        
        # 检查是否有改善
        if current_reward > self.best_reward + training_config.EARLY_STOPPING_MIN_DELTA:
            self.best_reward = current_reward
            self.patience_counter = 0
            
            # 保存最佳模型状态
            if training_config.EARLY_STOPPING_RESTORE_BEST:
                self.best_model_state = {
                    'actors': {agent_id: actor.state_dict() for agent_id, actor in self.actors.items()},
                    'critic': self.critic.state_dict(),
                    'actor_optimizers': {agent_id: opt.state_dict() for agent_id, opt in self.actor_optimizers.items()},
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'best_reward': self.best_reward
                }
            return False
        else:
            self.patience_counter += 1
            
        # 检查是否达到早停条件
        if self.patience_counter >= training_config.EARLY_STOPPING_PATIENCE:
            print(f"早停触发！最佳奖励: {self.best_reward:.2f}, 当前奖励: {current_reward:.2f}")
            
            # 恢复最佳模型
            if training_config.EARLY_STOPPING_RESTORE_BEST and self.best_model_state is not None:
                print("恢复最佳模型权重...")
                for agent_id, state_dict in self.best_model_state['actors'].items():
                    self.actors[agent_id].load_state_dict(state_dict)
                self.critic.load_state_dict(self.best_model_state['critic'])
                
            return True
        
        return False
    
    def update_lr_scheduler(self, current_reward: float = None):
        """更新学习率调度器"""
        from config import training_config
        
        if training_config.LR_SCHEDULER_TYPE == "plateau":
            # 平台调度器需要传入监控指标
            if current_reward is not None:
                for agent_id in self.actor_schedulers.keys():
                    self.actor_schedulers[agent_id].step(current_reward)
                self.critic_scheduler.step(current_reward)
        else:
            # 其他调度器按步数更新
            for agent_id in self.actor_schedulers.keys():
                self.actor_schedulers[agent_id].step()
            self.critic_scheduler.step()
    
    def get_current_lr(self) -> Dict[str, float]:
        """获取当前学习率"""
        actor_lrs = {}
        for agent_id, scheduler in self.actor_schedulers.items():
            actor_lrs[f'actor_{agent_id}'] = scheduler.get_last_lr()[0]
        
        critic_lr = self.critic_scheduler.get_last_lr()[0]
        
        return {
            'actor_lrs': actor_lrs,
            'critic_lr': critic_lr
        }

if __name__ == "__main__":
    # 测试MAPPO
    obs_dim = 36
    global_obs_dim = 54 + 100 * 23  # 假设100艘船
    action_dims = {i: 8 for i in range(10)}  # 假设10个智能体，每个8个动作
    
    agent = MAPPOAgent(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        action_dims=action_dims,
        device='cpu'
    )
    
    # 测试动作选择
    observations = {i: np.random.randn(obs_dim) for i in range(10)}
    actions, log_probs = agent.get_actions(observations)
    
    print(f"动作: {actions}")
    print(f"对数概率: {log_probs}")
    
    # 测试价值估计
    global_obs = np.random.randn(global_obs_dim)
    value = agent.get_value(global_obs)
    print(f"状态价值: {value}")
    
    print("MAPPO算法测试完成！")