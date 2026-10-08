"""
奖励函数优化器
实现多种奖励优化技术来提高训练收敛性，包括：
1. 动态奖励缩放
2. 奖励塑形 (Reward Shaping)
3. 奖励归一化
4. 渐进式奖励调整
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import math


class RewardOptimizer:
    """奖励优化器 - 提供多种奖励优化策略"""
    
    def __init__(self, 
                 strategy: str = 'adaptive_scaling',
                 window_size: int = 100,
                 target_range: Tuple[float, float] = (-10.0, 10.0),
                 shaping_factor: float = 0.1,
                 progress_bonus_weight: float = 0.05):
        """
        Args:
            strategy: 优化策略 ('adaptive_scaling', 'reward_shaping', 'progressive', 'hybrid')
            window_size: 滑动窗口大小
            target_range: 目标奖励范围
            shaping_factor: 奖励塑形因子
            progress_bonus_weight: 进步奖励权重
        """
        self.strategy = strategy
        self.window_size = window_size
        self.target_range = target_range
        self.shaping_factor = shaping_factor
        self.progress_bonus_weight = progress_bonus_weight
        
        # 统计信息
        self.reward_history = deque(maxlen=window_size)
        self.baseline_history = deque(maxlen=window_size)
        self.iteration_count = 0
        
        # 动态缩放参数
        self.current_scale = 1.0
        self.scale_momentum = 0.99
        self.min_scale = 0.01
        self.max_scale = 10.0
        
        # 奖励塑形状态
        self.previous_metrics = {}
        self.best_metrics = {}
        
        # 渐进式调整参数
        self.warmup_iterations = 200
        self.convergence_threshold = 0.01
        
    def optimize_rewards(self, 
                        raw_rewards: Dict[int, float],
                        ship_metrics: Dict[int, Dict] = None,
                        global_metrics: Dict = None) -> Dict[int, float]:
        """
        优化奖励函数
        
        Args:
            raw_rewards: 原始奖励
            ship_metrics: 船舶级别的指标
            global_metrics: 全局指标
            
        Returns:
            优化后的奖励
        """
        if not raw_rewards:
            return raw_rewards
            
        self.iteration_count += 1
        
        if self.strategy == 'adaptive_scaling':
            return self._adaptive_scaling(raw_rewards)
        elif self.strategy == 'reward_shaping':
            return self._reward_shaping(raw_rewards, ship_metrics, global_metrics)
        elif self.strategy == 'progressive':
            return self._progressive_adjustment(raw_rewards, ship_metrics)
        elif self.strategy == 'hybrid':
            return self._hybrid_optimization(raw_rewards, ship_metrics, global_metrics)
        else:
            return raw_rewards
    
    def _adaptive_scaling(self, rewards: Dict[int, float]) -> Dict[int, float]:
        """修复的自适应奖励缩放 - 保持奖励符号正确"""
        reward_values = list(rewards.values())
        if not reward_values:
            return rewards
            
        # 更新历史记录
        self.reward_history.extend(reward_values)
        
        if len(self.reward_history) < 20:  # 最小历史要求
            # 历史数据不足时，只进行轻微缩放，保持符号
            scaled_rewards = {}
            for ship_id, reward in rewards.items():
                # 简单缩放，保持符号不变
                scaled_rewards[ship_id] = reward * 0.8  # 轻微缩放
            return scaled_rewards
            
        # 使用历史数据计算统计量
        historical_values = list(self.reward_history)
        hist_std = np.std(historical_values)
        
        # 只进行标准差缩放，不改变均值
        if hist_std > 0:
            # 目标标准差：减少波动但不过度压缩
            target_std = min(hist_std * 0.7, 50.0)  # 保持70%的标准差，最大50
            scale_factor = target_std / hist_std
            
            # 限制缩放因子变化，增加稳定性
            if hasattr(self, 'last_scale_factor'):
                max_change = 0.05  # 每次最多变化5%
                scale_factor = np.clip(scale_factor, 
                                     self.last_scale_factor * (1 - max_change),
                                     self.last_scale_factor * (1 + max_change))
            
            # 确保缩放因子在合理范围内
            scale_factor = np.clip(scale_factor, 0.5, 1.5)
            self.last_scale_factor = scale_factor
        else:
            scale_factor = 1.0
            self.last_scale_factor = 1.0
        
        # 应用缩放 - 只缩放幅度，不改变符号
        scaled_rewards = {}
        current_mean = np.mean(reward_values)
        
        for ship_id, reward in rewards.items():
            # 相对于当前均值进行缩放，保持符号
            deviation = reward - current_mean
            scaled_deviation = deviation * scale_factor
            scaled_reward = current_mean + scaled_deviation
            
            # 额外检查：确保符号不变
            if (reward > 0 and scaled_reward < 0) or (reward < 0 and scaled_reward > 0):
                # 如果符号改变了，使用简单缩放
                scaled_reward = reward * scale_factor
            
            scaled_rewards[ship_id] = scaled_reward
            
        return scaled_rewards
    
    def _reward_shaping(self, 
                       rewards: Dict[int, float],
                       ship_metrics: Dict[int, Dict] = None,
                       global_metrics: Dict = None) -> Dict[int, float]:
        """奖励塑形 - 添加中间奖励信号"""
        shaped_rewards = rewards.copy()
        
        if ship_metrics is None:
            return shaped_rewards
            
        for ship_id, base_reward in rewards.items():
            if ship_id not in ship_metrics:
                continue
                
            metrics = ship_metrics[ship_id]
            shaping_bonus = 0.0
            
            # 1. 排放减少奖励
            if 'emissions' in metrics and ship_id in self.previous_metrics:
                prev_emissions = self.previous_metrics[ship_id].get('emissions', metrics['emissions'])
                emission_reduction = prev_emissions - metrics['emissions']
                if emission_reduction > 0:
                    shaping_bonus += emission_reduction * 0.001  # 每吨CO2减少给予0.001奖励
            
            # 2. 成本效率奖励
            if 'fuel_cost' in metrics and 'emissions' in metrics:
                if metrics['emissions'] > 0:
                    cost_efficiency = metrics['fuel_cost'] / metrics['emissions']
                    # 奖励低成本高效率
                    if ship_id in self.previous_metrics:
                        prev_efficiency = self.previous_metrics[ship_id].get('cost_efficiency', cost_efficiency)
                        if cost_efficiency < prev_efficiency:
                            shaping_bonus += (prev_efficiency - cost_efficiency) * 0.1
            
            # 3. 改造进度奖励
            if 'retrofit_progress' in metrics:
                progress = metrics['retrofit_progress']
                if progress > 0:
                    shaping_bonus += progress * 0.05  # 改造进度奖励
            
            # 4. 合规性奖励
            if 'compliance_score' in metrics:
                compliance = metrics['compliance_score']
                if compliance > 0.8:  # 高合规性奖励
                    shaping_bonus += (compliance - 0.8) * 0.2
            
            # 应用塑形奖励
            shaped_rewards[ship_id] = base_reward + shaping_bonus * self.shaping_factor
            
        # 更新历史指标
        self.previous_metrics = ship_metrics.copy()
        
        return shaped_rewards
    
    def _progressive_adjustment(self, 
                              rewards: Dict[int, float],
                              ship_metrics: Dict[int, Dict] = None) -> Dict[int, float]:
        """渐进式奖励调整"""
        if self.iteration_count <= self.warmup_iterations:
            # 预热期：使用较小的奖励缩放
            warmup_factor = self.iteration_count / self.warmup_iterations
            scale_factor = 0.1 + 0.9 * warmup_factor
        else:
            # 正常期：根据收敛情况调整
            if len(self.reward_history) >= self.window_size:
                recent_rewards = list(self.reward_history)[-50:]
                reward_std = np.std(recent_rewards)
                
                if reward_std < self.convergence_threshold:
                    # 收敛时增加探索
                    scale_factor = 1.2
                else:
                    # 未收敛时稳定训练
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
        
        # 应用渐进式调整
        adjusted_rewards = {}
        for ship_id, reward in rewards.items():
            adjusted_rewards[ship_id] = reward * scale_factor
            
        return adjusted_rewards
    
    def _hybrid_optimization(self, 
                           rewards: Dict[int, float],
                           ship_metrics: Dict[int, Dict] = None,
                           global_metrics: Dict = None) -> Dict[int, float]:
        """混合优化策略"""
        # 1. 首先应用自适应缩放
        scaled_rewards = self._adaptive_scaling(rewards)
        
        # 2. 然后应用奖励塑形
        shaped_rewards = self._reward_shaping(scaled_rewards, ship_metrics, global_metrics)
        
        # 3. 最后应用渐进式调整
        final_rewards = self._progressive_adjustment(shaped_rewards, ship_metrics)
        
        return final_rewards
    
    def get_optimization_stats(self) -> Dict:
        """获取优化统计信息"""
        stats = {
            'strategy': self.strategy,
            'iteration_count': self.iteration_count,
            'current_scale': self.current_scale,
            'reward_history_size': len(self.reward_history)
        }
        
        if self.reward_history:
            recent_rewards = list(self.reward_history)[-50:] if len(self.reward_history) >= 50 else list(self.reward_history)
            stats.update({
                'recent_mean': np.mean(recent_rewards),
                'recent_std': np.std(recent_rewards),
                'recent_min': np.min(recent_rewards),
                'recent_max': np.max(recent_rewards)
            })
            
        return stats


class RewardSmoother:
    """奖励平滑器 - 减少奖励波动"""
    
    def __init__(self, window_size: int = 10, smoothing_factor: float = 0.3):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.reward_buffer = {}  # 每个智能体的奖励缓冲区
        self.smoothed_rewards = {}  # 平滑后的奖励
        
    def smooth_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        """修复的奖励平滑 - 保持奖励符号正确"""
        smoothed = {}
        
        for ship_id, reward in rewards.items():
            # 初始化缓冲区
            if ship_id not in self.reward_buffer:
                self.reward_buffer[ship_id] = deque(maxlen=self.window_size)
                self.smoothed_rewards[ship_id] = reward
            
            # 添加新奖励到缓冲区
            self.reward_buffer[ship_id].append(reward)
            
            # 计算移动平均
            if len(self.reward_buffer[ship_id]) >= 3:
                moving_avg = np.mean(list(self.reward_buffer[ship_id]))
                
                # 使用指数移动平均进行平滑，但保持符号
                ema_smoothed = (
                    self.smoothing_factor * reward + 
                    (1 - self.smoothing_factor) * self.smoothed_rewards[ship_id]
                )
                
                # 结合移动平均和指数平滑
                combined_smoothed = 0.7 * ema_smoothed + 0.3 * moving_avg
                
                # 符号检查：确保平滑后的奖励符号与原始奖励一致
                if (reward > 0 and combined_smoothed < 0) or (reward < 0 and combined_smoothed > 0):
                    # 如果符号改变，使用更保守的平滑
                    combined_smoothed = 0.9 * reward + 0.1 * self.smoothed_rewards[ship_id]
                
                self.smoothed_rewards[ship_id] = ema_smoothed
                smoothed[ship_id] = combined_smoothed
            else:
                # 数据不足时使用简单平滑，保持符号
                simple_smoothed = 0.8 * reward + 0.2 * self.smoothed_rewards[ship_id]
                
                # 符号检查
                if (reward > 0 and simple_smoothed < 0) or (reward < 0 and simple_smoothed > 0):
                    simple_smoothed = reward  # 保持原始值
                
                self.smoothed_rewards[ship_id] = simple_smoothed
                smoothed[ship_id] = simple_smoothed
        
        return smoothed


class AdvancedRewardProcessor:
    """高级奖励处理器 - 集成奖励优化器和平滑器"""
    
    def __init__(self, 
                 optimization_strategy: str = 'adaptive_scaling',
                 clip_range: float = 20.0,
                 normalization: bool = True,
                 enable_smoothing: bool = True,
                 smoothing_window: int = 8,
                 smoothing_factor: float = 0.2):
        """
        Args:
            optimization_strategy: 优化策略
            clip_range: 裁剪范围
            normalization: 是否启用归一化
            enable_smoothing: 是否启用奖励平滑
            smoothing_window: 平滑窗口大小
            smoothing_factor: 平滑因子
        """
        self.clip_range = clip_range
        self.normalization = normalization
        
        # 奖励优化器
        self.optimizer = RewardOptimizer(
            strategy=optimization_strategy,
            target_range=(-clip_range/2, clip_range/2)
        )
        
        # 归一化器
        if normalization:
            self.normalizer = RewardNormalizer()
        else:
            self.normalizer = None
            
        # 平滑器
        if enable_smoothing:
            self.smoother = RewardSmoother(
                window_size=smoothing_window,
                smoothing_factor=smoothing_factor
            )
        else:
            self.smoother = None
            
        # 统计信息
        self.total_rewards = 0.0
        self.reward_count = 0
        self.update_count = 0
    
    def process_rewards(self, 
                       rewards: Dict[int, float],
                       ship_metrics: Dict[int, Dict] = None,
                       global_metrics: Dict = None) -> Dict[int, float]:
        """处理奖励 - 包含优化、平滑、归一化和裁剪"""
        if not rewards:
            return rewards
            
        # 1. 奖励优化
        optimized_rewards = self.optimizer.optimize_rewards(rewards, ship_metrics, global_metrics)
        
        # 2. 奖励平滑（新增）
        if self.smoother:
            smoothed_rewards = self.smoother.smooth_rewards(optimized_rewards)
        else:
            smoothed_rewards = optimized_rewards
        
        # 3. 归一化（可选）
        if self.normalizer:
            reward_values = np.array(list(smoothed_rewards.values()))
            self.normalizer.update(reward_values)
            normalized_values = self.normalizer.normalize(reward_values)
            
            normalized_rewards = {}
            for i, ship_id in enumerate(smoothed_rewards.keys()):
                normalized_rewards[ship_id] = normalized_values[i]
        else:
            normalized_rewards = smoothed_rewards
        
        # 4. 裁剪
        final_rewards = {}
        for ship_id, reward in normalized_rewards.items():
            final_rewards[ship_id] = np.clip(reward, -self.clip_range, self.clip_range)
        
        # 更新统计
        self._update_statistics(final_rewards)
        
        return final_rewards
    
    def _update_statistics(self, rewards: Dict[int, float]):
        """更新统计信息"""
        reward_values = list(rewards.values())
        self.total_rewards += sum(reward_values)
        self.reward_count += len(reward_values)
        self.update_count += 1
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_reward = self.total_rewards / max(1, self.reward_count)
        
        stats = {
            'clip_range': self.clip_range,
            'avg_reward': avg_reward,
            'update_count': self.update_count,
            'total_processed': self.reward_count,
            'normalization_enabled': self.normalization is not None
        }
        
        # 添加优化器统计
        optimizer_stats = self.optimizer.get_optimization_stats()
        stats.update({f'optimizer_{k}': v for k, v in optimizer_stats.items()})
        
        return stats


class RewardNormalizer:
    """奖励归一化器"""
    
    def __init__(self, eps: float = 1e-8, momentum: float = 0.99):
        self.eps = eps
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, rewards: np.ndarray):
        """更新统计信息"""
        if len(rewards) == 0:
            return
            
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = max(batch_var, 1e-3)
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            self.var = max(self.var, 1e-3)
        
        self.count += len(rewards)
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """归一化奖励 - 修复版本，保持奖励的相对大小"""
        if len(rewards) == 0:
            return rewards
            
        # 使用更温和的归一化方式，避免将奖励压缩到接近0
        std = np.sqrt(self.var + self.eps)
        
        # 如果方差太小，说明奖励变化不大，直接返回原始奖励
        if std < 0.1:
            return rewards
            
        # 使用缩放而不是标准化，保持奖励的符号和相对大小
        scale_factor = min(10.0 / std, 1.0)  # 限制缩放因子
        return rewards * scale_factor