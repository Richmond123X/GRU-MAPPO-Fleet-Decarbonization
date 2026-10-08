"""集成式训练文件 - 支持多种算法训练海运脱碳系统"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import json
import argparse
import copy
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# 导入环境和算法
from environment import MaritimeDecarbonizationEnv
from mappo import MAPPOAgent
from qmix import QMIXAgent  # 新增QMIX算法导入
from load_real_ships import load_ships_from_csv
# from adaptive_scheduler import AdaptiveParameterScheduler, MultiStrategyScheduler, create_adaptive_scheduler  # 暂时注释掉
from config import fuel_config, policy_config, sys_config, get_discount_factor, quarter_to_year

class ObservationNormalizer:
    """观察值标准化器 - 实现在线均值和方差计算"""
    
    def __init__(self, obs_dim: int, eps: float = 1e-8):
        self.obs_dim = obs_dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
        self.std = np.ones(obs_dim, dtype=np.float32)
    
    def update(self, obs: np.ndarray):
        """更新统计信息 - 使用Welford算法进行在线更新"""
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        self.std = np.sqrt(self.var + self.eps)
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """标准化观察值"""
        if self.count == 0:
            return obs
        return (obs - self.mean) / self.std
    
    def denormalize(self, normalized_obs: np.ndarray) -> np.ndarray:
        """反标准化观察值"""
        if self.count == 0:
            return normalized_obs
        return normalized_obs * self.std + self.mean

class RewardNormalizer:
    """奖励标准化器 - 使用移动平均和标准差来标准化奖励，提高训练稳定性"""
    
    def __init__(self, eps: float = 1e-8, momentum: float = 0.995):
        self.eps = eps
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, rewards: np.ndarray):
        """更新奖励统计信息"""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = max(batch_var, 1e-3)  # 防止方差过小
        else:
            # 使用更稳定的移动平均更新
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            self.var = max(self.var, 1e-3)  # 确保方差不会过小
        
        self.count += len(rewards)
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """标准化奖励"""
        std = np.sqrt(self.var + self.eps)
        return (rewards - self.mean) / std
    
    def get_stats(self) -> tuple:
        """获取当前统计信息"""
        return self.mean, np.sqrt(self.var + self.eps)


class ImprovedRewardProcessor:
    """简化的奖励处理器 - 仅使用合理裁剪"""
    
    def __init__(self, 
                 clip_range: float = 50.0,
                 scale_factor: float = 1.0):
        """
        Args:
            clip_range: 奖励裁剪范围（10万美元单位）
            scale_factor: 奖励缩放因子
        """
        self.clip_range = clip_range
        self.scale_factor = scale_factor
        
        # 简单统计信息
        self.update_count = 0
        self.total_rewards = 0.0
        self.reward_count = 0
        
    def update_statistics(self, rewards: Dict[int, float]):
        """更新简单统计信息"""
        if not rewards:
            return
            
        reward_values = list(rewards.values())
        self.total_rewards += sum(reward_values)
        self.reward_count += len(reward_values)
        self.update_count += 1
    
    def adaptive_scale_update(self):
        """简化的缩放更新（不执行任何操作）"""
        pass
    
    def process_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        """简单的奖励处理：保持原始百万单位并进行合理裁剪"""
        if not rewards:
            return rewards
            
        # 更新统计信息
        self.update_statistics(rewards)
        
        processed_rewards = {}
        
        for ship_id, reward in rewards.items():
            # 保持原始奖励，不再缩小十倍
            scaled_reward = reward * self.scale_factor
            # 然后裁剪（奖励已经是百万美元单位）
            clipped_reward = np.clip(scaled_reward, -self.clip_range, self.clip_range)
            
            processed_rewards[ship_id] = clipped_reward
        
        return processed_rewards
    
    def get_stats(self) -> Dict[str, float]:
        """获取处理器统计信息"""
        avg_reward = self.total_rewards / max(1, self.reward_count) if self.reward_count > 0 else 0.0
        
        return {
            'clip_range': self.clip_range,
            'scale_factor': self.scale_factor,
            'avg_reward': avg_reward,
            'update_count': self.update_count,
            'total_processed': self.reward_count
        }

def create_global_observation(env: MaritimeDecarbonizationEnv) -> np.ndarray:
    """创建简化的全局观察状态（提高性能）"""
    # 使用环境内置的全局观察方法
    return env.get_global_observation()

def plot_training_curves(history: Dict[str, List[float]], algorithm_name: str, save_dir: str = None):
    """绘制训练曲线"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{algorithm_name}训练曲线', fontsize=16)
    
    # 总排放量
    axes[0, 0].plot(history['iterations'], history['total_emissions'])
    axes[0, 0].set_title('总排放量')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('排放量 (吨CO2)')
    axes[0, 0].grid(True)
    
    # 总成本
    axes[0, 1].plot(history['iterations'], history['total_costs'])
    axes[0, 1].set_title('总成本')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('成本 (美元)')
    axes[0, 1].grid(True)
    
    # 总奖励 - 修复显示问题
    rewards_array = np.array(history['total_rewards'])
    axes[0, 2].plot(history['iterations'], rewards_array)
    axes[0, 2].set_title('总奖励（处理后）')
    axes[0, 2].set_xlabel('迭代次数')
    axes[0, 2].set_ylabel('奖励')
    axes[0, 2].grid(True)
    
    # 添加奖励趋势线
    if len(rewards_array) > 1:
        z = np.polyfit(history['iterations'], rewards_array, 1)
        p = np.poly1d(z)
        axes[0, 2].plot(history['iterations'], p(history['iterations']), "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.3f})')
        axes[0, 2].legend()
    
    # Actor损失
    if 'actor_losses' in history and history['actor_losses']:
        axes[1, 0].plot(history['iterations'], history['actor_losses'])
        axes[1, 0].set_title('Actor损失')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].grid(True)
    
    # Critic损失
    if 'critic_losses' in history and history['critic_losses']:
        axes[1, 1].plot(history['iterations'], history['critic_losses'])
        axes[1, 1].set_title('Critic损失')
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True)
    
    # 熵
    if 'entropies' in history and history['entropies']:
        axes[1, 2].plot(history['iterations'], history['entropies'])
        axes[1, 2].set_title('策略熵')
        axes[1, 2].set_xlabel('迭代次数')
        axes[1, 2].set_ylabel('熵')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # 根据保存目录决定图片保存路径
    if save_dir and os.path.exists(save_dir):
        plot_path = f'{save_dir}/{algorithm_name}_training_curves.png'
    else:
        plot_path = f'{algorithm_name}_training_curves.png'
    
    plt.savefig(plot_path)
    plt.close()

def plot_normalized_metrics(history: Dict[str, List[float]], algorithm_name: str, reward_processor=None, save_dir: str = None):
    """绘制缩放后的奖励、成本及排放量"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    iterations = np.array(history['iterations'])
    emissions = np.array(history['total_emissions'])
    costs = np.array(history['total_costs'])
    rewards = np.array(history['total_rewards'])
    
    # 缩放因子
    if reward_processor is not None:
        # 使用实际的奖励处理缩放因子
        reward_stats = reward_processor.get_stats()
        # 检查是否有scale_factor，如果没有则使用默认值
        REWARD_SCALE = reward_stats.get('scale_factor', reward_stats.get('optimizer_current_scale', 1.0))
        reward_scale_label = f"×{REWARD_SCALE:.2e}，实际训练缩放"
    else:
        # 如果没有reward_processor，使用默认的可视化缩放因子
        REWARD_SCALE = 1e-8
        reward_scale_label = "×1e-8，仅用于显示"
    
    COST_SCALE = 1e-9    # 成本缩放因子，将十亿级成本缩放到合理范围
    EMISSION_SCALE = 1e-6  # 排放量缩放因子，转换为百万吨
    
    # 应用缩放
    scaled_rewards = rewards * REWARD_SCALE
    scaled_costs = costs * COST_SCALE
    scaled_emissions = emissions * EMISSION_SCALE
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{algorithm_name} - 训练指标分析', fontsize=16)
    
    # 子图1: 实际奖励
    axes[0, 0].plot(iterations, rewards, label='实际奖励', color='blue', linewidth=2)
    axes[0, 0].set_title('实际奖励')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('奖励值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 实际成本
    axes[0, 1].plot(iterations, costs, label='实际成本', color='red', linewidth=2)
    axes[0, 1].set_title('实际成本')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('成本值 (美元)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 实际排放量
    axes[1, 0].plot(iterations, emissions, label='实际排放量', color='green', linewidth=2)
    axes[1, 0].set_title('实际排放量')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('排放量 (吨CO2)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 综合指标对比
    # 为了在同一图中显示，需要标准化到相似范围
    norm_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min()) if rewards.max() != rewards.min() else np.zeros_like(rewards)
    norm_costs = (costs - costs.min()) / (costs.max() - costs.min()) if costs.max() != costs.min() else np.zeros_like(costs)
    norm_emissions = (emissions - emissions.min()) / (emissions.max() - emissions.min()) if emissions.max() != emissions.min() else np.zeros_like(emissions)
    
    axes[1, 1].plot(iterations, norm_rewards, label='标准化实际奖励', color='blue', alpha=0.8)
    axes[1, 1].plot(iterations, norm_costs, label='标准化实际成本', color='red', alpha=0.8)
    axes[1, 1].plot(iterations, norm_emissions, label='标准化实际排放量', color='green', alpha=0.8)
    axes[1, 1].set_title('标准化指标对比')
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('标准化值 [0,1]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 根据保存目录决定图片保存路径
    if save_dir and os.path.exists(save_dir):
        plot_path = f'{save_dir}/{algorithm_name}_scaled_metrics.png'
    else:
        plot_path = f'{algorithm_name}_scaled_metrics.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n=== {algorithm_name} 缩放指标统计 ===")
    print(f"原始奖励范围: {rewards.min():.2f} - {rewards.max():.2f}")
    
    # 显示实际奖励处理缩放范围（如果提供了reward_processor）
    if reward_processor is not None:
        reward_stats = reward_processor.get_stats()
        actual_scale_factor = reward_stats.get('scale_factor', reward_stats.get('optimizer_current_scale', 1.0))
        actual_scaled_min = rewards.min() * actual_scale_factor
        actual_scaled_max = rewards.max() * actual_scale_factor
        print(f"实际奖励处理缩放范围: {actual_scaled_min:.4f} - {actual_scaled_max:.4f} (×{actual_scale_factor:.2e}，用于训练)")
    
    print(f"图表缩放奖励范围: {scaled_rewards.min():.6f} - {scaled_rewards.max():.6f} ({reward_scale_label})")
    print(f"原始成本范围: {costs.min():.2f} - {costs.max():.2f} 美元")
    print(f"可视化缩放成本范围: {scaled_costs.min():.4f} - {scaled_costs.max():.4f} (×1e-9，仅用于图表显示)")
    print(f"原始排放量范围: {emissions.min():.2f} - {emissions.max():.2f} 吨CO2")
    print(f"缩放排放量范围: {scaled_emissions.min():.4f} - {scaled_emissions.max():.4f} 百万吨CO2")
    print(f"缩放指标图表已保存: {plot_path}")

def train(algorithm: str = 'mappo',
         num_iterations: int = 1000, 
         num_ships: int = 100,
         random_seed: int = 42,
         save_interval: int = 100,
         device: str = 'auto',
         use_adaptive_scheduler: bool = True,
         scheduler_strategy: str = 'hybrid',
         use_real_ships: bool = True,
         ships_csv_path: str = 'Container_ship.xlsx',
         load_model: str = None,  # 新增：加载已保存的模型路径
         start_iteration: int = 0,  # 新增：起始迭代序号，用于继续训练时的模型保存
         save_dir: str = None,  # 新增：自定义保存目录路径
         # GPU优化参数
         gpu_optimized: bool = False,
         batch_size_multiplier: float = 1.0,
         memory_fraction: float = 0.9,
         prefetch_factor: int = 2,
         use_mixed_precision: bool = False,
         pin_memory: bool = False,
         non_blocking_transfer: bool = False,
         vectorized_env: bool = False,
         gpu_env_batch_size: int = 16,
         async_env_steps: bool = False,
         data_prefetch: bool = False,
         tensor_caching: bool = False,
         batch_data_transfer: bool = False,
         fused_operations: bool = False,
         gradient_checkpointing: bool = False,
         compile_model: bool = False,
         bio_cap_gj: float = None,
         bio_cap_ratio: float = None,
         **kwargs):
    """集成训练函数 - 支持多种算法"""
    
    # 自动检测并设置最佳设备
    # 设备配置
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device in ['cuda', 'gpu']:
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            device = 'cpu'
    
    # CPU优化配置
    if device == 'cpu':
        cpu_count = os.cpu_count() or 1
        cap_threads = max(1, int(cpu_count * 0.7))
        env_caps = []
        for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            v = os.environ.get(k)
            if v and v.isdigit():
                env_caps.append(int(v))
        env_cap = min(env_caps) if env_caps else cap_threads
        num_threads = min(env_cap, cap_threads, 16)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        torch.set_num_threads(num_threads)
        
        if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
        
        torch.jit.set_fusion_strategy([('STATIC', 10), ('DYNAMIC', 10)])
        torch.backends.mkldnn.verbose = 0
        
        print(f"CPU性能优化完成，使用 {num_threads} 个线程")
    
    # GPU优化配置
    elif device == 'cuda':
        use_mixed_precision = True
        if use_mixed_precision:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        memory_fraction = 0.55
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        print("  预期性能提升: 2-4倍")
    
    print(f"开始{algorithm.upper()}训练 ({num_iterations}次迭代)")
    
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 训练超参数
    OBSERVATION_CLIP = 3.0  # 观察值裁剪范围，进一步减小以降低波动
    REWARD_CLIP = 5.0  # 奖励值裁剪范围，减小裁剪范围
    
    # 创建高级奖励处理器 - 使用优化策略，启用平滑功能
    from reward_optimizer import AdvancedRewardProcessor
    reward_processor = AdvancedRewardProcessor(
        optimization_strategy='adaptive_scaling',  # 仅使用自适应缩放
        clip_range=25.0,  # 适中的裁剪范围
        normalization=False,  # 暂时禁用归一化
        enable_smoothing=True,  # 启用奖励平滑
        smoothing_window=8,     # 平滑窗口大小
        smoothing_factor=0.2    # 平滑因子
    )
    
    # 创建环境 - 现在必须使用真实船舶数据
    if use_real_ships and ships_csv_path:
        # 加载真实船舶数据，使用指定数量的船舶
        if ships_csv_path.endswith('.xlsx'):
            from load_real_ships import load_ships_from_excel
            # 传递num_ships参数以使用指定数量的船舶
            real_fleet = load_ships_from_excel(ships_csv_path, num_ships)
            print(f"已加载Container_ship数据集，共{len(real_fleet.ships)}艘船舶")
        else:
            real_fleet = load_ships_from_csv(ships_csv_path, num_ships)
        
        # 打印实际使用的船舶数量
        actual_num_ships = len(real_fleet.ships)
        print(f"随机种子:{random_seed} | 设备:{device} | 船舶:{actual_num_ships}艘")
        
        # 创建环境并传入真实船队
        env = MaritimeDecarbonizationEnv(
            num_ships=len(real_fleet.ships), 
            random_seed=random_seed, 
            fleet=real_fleet,
            bio_supply_cap_per_quarter=bio_cap_gj,
            bio_supply_ratio=bio_cap_ratio,
            bio_supply_base_gj=kwargs.get('bio_cap_base_gj')
        )
    else:
        raise ValueError("必须提供真实船舶数据，请设置 --use_real_ships 和 --ships_csv_path 参数")
    
    # 获取观察和动作空间维度
    obs_dim = 36  # 单个智能体观察维度
    
    # 获取实际的全局观察维度
    temp_global_obs = create_global_observation(env)
    global_obs_dim = len(temp_global_obs)
    
    # 初始化标准化器
    obs_normalizer = ObservationNormalizer(obs_dim)
    global_obs_normalizer = ObservationNormalizer(global_obs_dim)
    
    # 创建动作空间字典
    action_dims = {}
    for ship in env.fleet.ships:
        if ship.ship_type.value == 'F':
            action_dims[ship.ship_id] = 10
        elif ship.ship_type.value in ['G', 'FG']:
            action_dims[ship.ship_id] = 5
        elif ship.ship_type.value == 'Z_DF':
            action_dims[ship.ship_id] = 1
    
    # 统计船队信息
    f_ships = sum(1 for dim in action_dims.values() if dim == 10)
    g_ships = sum(1 for dim in action_dims.values() if dim == 5)
    z_ships = sum(1 for dim in action_dims.values() if dim == 1)
    
    print(f"船队配置: F{f_ships}艘 | G/FG{g_ships}艘 | Z-DF{z_ships}艘")
    
    # 根据设备类型、GPU内存和船舶数量动态调整批次大小
    if device == 'cuda':
        # 根据GPU内存和船舶数量动态调整批次大小
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # 基础批次大小配置 - 极度内存优化避免OOM
        if gpu_memory_gb >= 8:
            base_mappo_batch = 256  # 8GB GPU极度降低批次大小
            gpu_config = "8GB+GPU配置 - 极度内存优化"
        elif gpu_memory_gb >= 3.8:
            base_mappo_batch = 128  # 4GB GPU极度降低批次大小
            gpu_config = "4GB GPU配置 - 极度内存优化"
        else:
            base_mappo_batch = 64   # 小内存GPU极度降低
            gpu_config = "小内存GPU配置 - 极度内存优化"
        
        # 根据船舶数量调整批次大小 - 平衡GPU使用率和训练速度
        # 对于100艘船，适度增加批次大小
        if num_ships >= 100:
            # 100艘船适度增加批次大小，平衡GPU利用和训练速度
            ship_scale_factor = 1.0  # 保持标准批次大小
        elif num_ships >= 75:
            ship_scale_factor = 1.0  # 标准批次大小
        else:
            ship_scale_factor = 0.8  # 小规模训练减少批次大小
            
        mappo_batch_size = max(64, int(base_mappo_batch * ship_scale_factor * batch_size_multiplier))
        
        # 通信算法优化：适度增加批次大小平衡性能和稳定性
        if algorithm.lower() == 'mappo-communication':
            if num_ships >= 100:
                # 100艘船通信算法：适度增加批次大小，避免内存问题
                comm_scale_factor = 1.2  # 增加20%批次大小
                mappo_batch_size = max(768, int(mappo_batch_size * comm_scale_factor))
            else:
                comm_scale_factor = 1.1  # 中等规模适度增加
                mappo_batch_size = max(384, int(mappo_batch_size * comm_scale_factor))
        
        ppo_epochs = 15  # 适中的更新轮数
    else:
        # CPU训练：激进优化批次大小以利用多核性能
        cpu_cores = os.cpu_count()
        
        # 激进优化CPU配置 - 大幅提升批次大小和并行度
        if cpu_cores >= 16:  # 高性能CPU
            base_mappo_batch = 1024  # 大幅增加批次大小，充分利用CPU缓存
            cpu_config = "高性能CPU激进优化配置"
        elif cpu_cores >= 8:  # 中等性能CPU
            base_mappo_batch = 768
            cpu_config = "中等性能CPU激进优化配置"
        else:  # 低性能CPU
            base_mappo_batch = 512
            cpu_config = "标准CPU激进优化配置"
        
        # 根据船舶数量调整批次大小（激进调整）
        if num_ships >= 100:
            ship_scale_factor = 1.5  # 大规模训练大幅增加批次，提升并行效率
        elif num_ships >= 75:
            ship_scale_factor = 1.3  # 标准批次大小
        else:
            ship_scale_factor = 1.0  # 小规模训练保持标准批次大小
        
        mappo_batch_size = int(base_mappo_batch * ship_scale_factor * batch_size_multiplier)
        
        # 增加PPO轮数，充分利用CPU计算能力
        ppo_epochs = 12  # 增加到12轮，提升训练效率
        
        # 通信算法CPU专用优化
        if algorithm.lower() == 'mappo-communication':
            if num_ships >= 100:
                comm_scale_factor = 1.8  # 增加80%批次大小
                mappo_batch_size = int(mappo_batch_size * comm_scale_factor)
            else:
                comm_scale_factor = 1.5  # 中等规模大幅增加
                mappo_batch_size = int(mappo_batch_size * comm_scale_factor)
    
    # 创建自适应参数调度器
    scheduler = None
    # if use_adaptive_scheduler:
    #     scheduler = create_adaptive_scheduler(
    #         strategy=scheduler_strategy,
    #         total_iterations=num_iterations,
    #         algorithm=algorithm
    #     )
    #     print(f"已创建自适应参数调度器: {scheduler_strategy}")
    
    # 标准化算法名称用于文件保存
    algorithm_normalized = algorithm.lower()
    
    # 根据选择的算法创建智能体
    if algorithm_normalized == 'mappo':
        agent = MAPPOAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            lr_actor=5e-5,      # 进一步降低Actor学习率，提高稳定性
            lr_critic=1e-4,     # 进一步降低Critic学习率，提高稳定性
            gamma=0.99,         # 标准折扣因子
            gae_lambda=0.95,    # 提高GAE lambda，增强稳定性
            clip_ratio=0.1,     # 进一步降低裁剪比例，减少策略更新幅度
            entropy_coef=0.01,  # 降低熵系数，减少随机性
            value_coef=0.5,     # 标准价值权重
            max_grad_norm=0.3,  # 更严格梯度裁剪：防止梯度爆炸
            ppo_epochs=6,       # 进一步减少更新轮数，避免过拟合
            mini_batch_size=mappo_batch_size,# 根据设备调整批次大小
            device=device
        )
    elif algorithm_normalized == 'gnn-mappo':
        from mappo_gnn import GNNMAPPOAgent
        agent = GNNMAPPOAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            lr_actor=5e-5,
            lr_critic=1e-4,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.3,
            ppo_epochs=6,
            mini_batch_size=mappo_batch_size,
            device=device,
            enable_gnn_coordination=True,
            hidden_dim=256
        )
    elif algorithm_normalized == 'gru-mappo':
        from gru_mappo import GRUMAPPOAgent
        agent = GRUMAPPOAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            sequence_length=10,  # GRU序列长度
            lr_actor=5e-5,
            lr_critic=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.3,
            ppo_epochs=6,
            mini_batch_size=mappo_batch_size,
            device=device,
            hidden_dim=256
        )
    elif algorithm_normalized == 'gru-mappo-communication':
        from gru_mappo_communication import GRUMAPPOCommunicationAgent
        agent = GRUMAPPOCommunicationAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            sequence_length=10,  # GRU序列长度
            lr_actor=5e-5,
            lr_critic=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.3,
            ppo_epochs=6,
            mini_batch_size=mappo_batch_size,
            device=device,
            hidden_dim=256,
            enable_communication=True,  # 启用通信机制
            num_comm_heads=4  # 通信注意力头数
        )
    elif algorithm_normalized == 'qmix':
        agent = QMIXAgent(
            fleet=env.fleet,  # 传入船队对象以获取动态动作空间
            obs_dim=obs_dim,
            global_state_dim=global_obs_dim,
            lr=1e-4,  # QMIX学习率
            gamma=0.99,
            epsilon=1.0,  # 初始探索率
            epsilon_min=0.05,  # 最小探索率
            epsilon_decay=0.995,  # 探索率衰减
            target_update_freq=200,  # 目标网络更新频率
            buffer_size=50000,  # 经验回放缓冲区大小
            batch_size=32,  # 训练批次大小
            device=device
        )
    elif algorithm_normalized == 'gru-mappo-gnn':
        from gru_mappo_gnn import GRUMAPPOGNNAgent
        agent = GRUMAPPOGNNAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            sequence_length=10,  # GRU序列长度
            lr_actor=5e-5,
            lr_critic=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.3,
            ppo_epochs=6,
            mini_batch_size=mappo_batch_size,
            device=device,
            hidden_dim=256,
            gnn_hidden_dim=128,  # GNN隐藏层维度
            num_heads=4,  # 多头注意力头数
            num_gnn_layers=2  # GNN层数
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}，目前支持: 'mappo', 'gnn-mappo', 'gru-mappo', 'gru-mappo-communication', 'gru-mappo-gnn'")
    
    # 加载已保存的模型（如果指定）
    if load_model:
        print(f"正在加载模型: {load_model}")
        try:
            if hasattr(agent, 'load_models'):
                agent.load_models(load_model)
            elif hasattr(agent, 'load_model'):
                agent.load_model(load_model)
            elif hasattr(agent, 'load'):
                agent.load(load_model)
            else:
                print(f"警告: {algorithm} 算法不支持模型加载")
            print(f"成功加载模型: {load_model}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将从头开始训练")
    
    # 训练记录
    training_history = {
        'iterations': [],
        'total_emissions': [],
        'total_costs': [],
        'total_rewards': [],
        'average_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'explained_variances': [],
        'ship_strategies': []  # 新增：保存每个迭代中每艘船的策略
    }
    
    # GPU内存优化设置 - 优化内存分配策略
    if device == 'cuda':
        # 清理GPU内存缓存
        torch.cuda.empty_cache()
        # 设置内存分配策略减少碎片化
        torch.cuda.set_per_process_memory_fraction(0.9)  # 限制内存使用避免OOM
        # 根据船舶数量调整内存分配策略 - 优化GPU使用率
        if num_ships >= 100:
            memory_fraction = 0.95  # 100艘船使用95%内存，最大化GPU利用
        elif num_ships >= 75:
            memory_fraction = 0.92  # 75艘船使用92%内存
        else:
            memory_fraction = 0.90  # 50艘船及以下使用90%内存
        
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # 强制预分配GPU内存以提高使用率 - 激进策略
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = int(total_memory * memory_fraction * 0.85)  # 预分配85%的目标内存，激进策略
        try:
            # 创建多个大张量强制分配内存
            dummy_tensors = []
            chunk_size = target_memory // 4  # 分4次分配大块内存
            for i in range(4):  # 分配全部4块
                tensor = torch.zeros(chunk_size // 4, dtype=torch.float32, device='cuda')
                dummy_tensors.append(tensor)
            # 保持张量一段时间强制GPU分配
            torch.cuda.synchronize()
            # 清理张量
            for tensor in dummy_tensors:
                del tensor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            pass  # 如果内存不足，忽略错误
    
    # 开始训练
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iteration_start_time = time.time()
        
        # 重置环境
        observations = env.reset()
        global_obs = create_global_observation(env)
        
        episode_rewards = {ship_id: 0.0 for ship_id in action_dims.keys()}
        episode_steps = 0
        episode_strategies = []  # 新增：收集整个episode的策略
        
        # 运行一个episode
        max_episode_length = getattr(agent, 'max_episode_length', 50)
        while not env.done and episode_steps < max_episode_length:
            # 更新观察值统计并标准化
            normalized_observations = {}
            for ship_id, obs in observations.items():
                # 更新统计信息
                obs_normalizer.update(obs)
                # 标准化观察值
                normalized_obs = obs_normalizer.normalize(obs)
                # 裁剪防止极值
                normalized_obs = np.clip(normalized_obs, -OBSERVATION_CLIP, OBSERVATION_CLIP)
                normalized_observations[ship_id] = normalized_obs
            
            # 更新全局观察统计并标准化
            global_obs_normalizer.update(global_obs)
            normalized_global_obs = global_obs_normalizer.normalize(global_obs)
            normalized_global_obs = np.clip(normalized_global_obs, -OBSERVATION_CLIP, OBSERVATION_CLIP)
            
            # 获取动作
            if algorithm.lower() == 'mappo-communication':
                actions, log_probs, value, comm_loss, coord_weights = agent.get_actions(normalized_observations, deterministic=False)
            elif algorithm.lower() == 'gnn-mappo':
                actions, log_probs = agent.get_actions(normalized_observations, env.fleet.ships, deterministic=False)
            elif algorithm.lower() == 'gru-mappo':
                actions, log_probs, values = agent.get_actions(normalized_observations, deterministic=False)
            elif algorithm.lower() == 'gru-mappo-communication':
                actions, log_probs, values = agent.get_actions(normalized_observations, deterministic=False, env=env)
            elif algorithm.lower() == 'gru-mappo-gnn':
                actions, log_probs, values = agent.get_actions(normalized_observations, env.fleet.ships, deterministic=False)
            elif algorithm.lower() == 'qmix':
                actions = agent.get_actions(normalized_observations, normalized_global_obs, training=True)
            else:  # mappo
                actions, log_probs = agent.get_actions(normalized_observations, deterministic=False)
            
            # 执行动作
            next_observations, rewards, dones, info = env.step(actions)
            next_global_obs = create_global_observation(env)
            
            # 收集当前步骤的船舶策略信息
            action_strings = env._convert_actions(actions)
            current_strategies = {}
            for ship in env.fleet.ships:
                ship_id = ship.ship_id
                action_code = actions.get(ship_id, 0)
                action_string = action_strings.get(ship_id, 'B0')
                
                current_strategies[ship_id] = {
                    'ship_type': ship.ship_type.name,
                    'teu_capacity': ship.teu_capacity,
                    'action_code': action_code,
                    'action_string': action_string,
                    'quarter': env.current_quarter,
                    'is_active': ship_id in [ship.ship_id for ship in env.fleet.get_active_ships()]
                }
            
            # 将当前步骤的策略添加到episode策略中
            episode_strategies.append({
                'step': episode_steps,
                'quarter': env.current_quarter,
                'strategies': current_strategies.copy()
            })
            
            # 供给约束监控
            if 'bio_supply_remaining' in info:
                try:
                    remaining = float(info['bio_supply_remaining'])
                    consumed = float(info.get('bio_supply_consumed', 0.0))
                    cap = remaining + consumed
                    if (iteration + 1) % 10 == 0:
                        print(f"    生物柴油供给: 已用 {consumed:,.1f} / 上限 {cap:,.1f} GJ | 剩余 {remaining:,.1f} GJ")
                except Exception:
                    pass

            # 使用改进的奖励处理器处理奖励
            processed_rewards = reward_processor.process_rewards(rewards)
            
            # 标准化下一步观察值
            normalized_next_observations = {}
            for ship_id, obs in next_observations.items():
                # 使用当前统计信息标准化（不更新统计）
                normalized_obs = obs_normalizer.normalize(obs)
                normalized_obs = np.clip(normalized_obs, -OBSERVATION_CLIP, OBSERVATION_CLIP)
                normalized_next_observations[ship_id] = normalized_obs
            
            # 标准化下一步全局观察
            normalized_next_global_obs = global_obs_normalizer.normalize(next_global_obs)
            normalized_next_global_obs = np.clip(normalized_next_global_obs, -OBSERVATION_CLIP, OBSERVATION_CLIP)
            
            # 存储经验
            if algorithm.lower() == 'mappo-communication':
                # MAPPO-Communication需要传递values、通信损失和协调权重
                agent.store_transition(
                    state=normalized_observations,
                    global_obs=normalized_global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=processed_rewards,
                    dones=dones,
                    values=value,
                    comm_loss=comm_loss,
                    coord_weights=coord_weights
                )
            elif algorithm.lower() == 'gru-mappo':
                # GRU-MAPPO需要传递values
                agent.store_transition(
                    observations=normalized_observations,
                    global_obs=normalized_global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=processed_rewards,
                    dones=dones,
                    values=values
                )
            elif algorithm.lower() == 'gru-mappo-communication':
                # GRU-MAPPO-Communication需要传递values
                agent.store_transition(
                    observations=normalized_observations,
                    global_obs=normalized_global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=processed_rewards,
                    dones=dones,
                    values=values
                )
            elif algorithm.lower() == 'gru-mappo-gnn':
                # GRU-MAPPO-GNN需要传递values
                agent.store_transition(
                    observations=normalized_observations,
                    global_obs=normalized_global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=processed_rewards,
                    dones=dones,
                    values=values
                )
            elif algorithm.lower() == 'qmix':
                # QMIX使用经验回放机制
                agent.store_experience(
                    observations=normalized_observations,
                    global_state=normalized_global_obs,
                    actions=actions,
                    rewards=processed_rewards,
                    next_observations=normalized_next_observations,
                    next_global_state=normalized_next_global_obs,
                    dones=dones
                )
            else:
                # 标准MAPPO调用
                agent.store_transition(
                    observations=normalized_observations,
                    global_obs=normalized_global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=processed_rewards,
                    dones=dones
                )
            
            # 累计奖励（使用处理后的奖励进行统计）
            for ship_id, processed_reward in processed_rewards.items():
                episode_rewards[ship_id] += processed_reward
            
            # 更新状态（使用归一化后的观察值）
            observations = normalized_next_observations
            global_obs = normalized_next_global_obs
            episode_steps += 1
        
        # 计算下一状态价值（用于GAE计算）
        if algorithm.lower() == 'mappo-communication':
            next_value = agent.get_value(observations) if not env.done else 0.0
        else:
            next_value = agent.get_value(global_obs) if not env.done else 0.0
        
        # 应用自适应参数调度
        if scheduler is not None:
            # 计算当前性能指标
            performance_metrics = {
                'total_reward': sum(processed_rewards.values()),
                'episode_length': episode_steps,
                'total_emissions': sum(ship.quarterly_emissions[-1] if ship.quarterly_emissions else 0 for ship in env.fleet.ships)
            }
            
            # 更新调度器并获取新参数
            new_params = scheduler.update(iteration, performance_metrics)
            
            # 应用新参数到智能体
            if hasattr(agent, 'update_hyperparameters'):
                agent.update_hyperparameters(new_params)
        
        # 更新网络 - 根据算法类型使用不同的参数
        if algorithm.lower() == 'mappo':
            update_stats = agent.update(next_value=next_value)
        elif algorithm.lower() == 'mappo-communication':
            update_stats = agent.update(next_value=next_value)
        elif algorithm.lower() == 'gru-mappo':
            update_stats = agent.update(next_value=next_value)
        elif algorithm.lower() == 'gru-mappo-communication':
            update_stats = agent.update(next_value=next_value)
        elif algorithm.lower() == 'gru-mappo-gnn':
            update_stats = agent.update(ships=env.fleet.ships, next_value=next_value)
        elif algorithm.lower() == 'qmix':
            # QMIX使用经验回放训练，不需要next_value
            update_stats = agent.train()
        else:
            update_stats = agent.update(next_value=next_value)
        
        # 如果使用FixedInitialMAPPOAgent，需要更新训练步数
        if hasattr(agent, 'step_training'):
            agent.step_training()
        
        # 计算统计信息
        # 计算当前季度的排放和各项成本/收益
        total_emissions = 0
        total_fuel_costs = 0
        total_retrofit_costs = 0
        total_opportunity_costs = 0
        total_compliance_costs = 0
        total_su_revenues = 0
        total_znz_rewards = 0
        
        # 统一单位转换因子：USD -> 百万USD（与环境保持一致）
        UNIT_SCALE = 1e-6  # 1百万美元 = 1,000,000美元
        
        for ship in env.fleet.ships:
            # 排放统计
            if ship.quarterly_emissions:
                total_emissions += ship.quarterly_emissions[-1]
            
            # 燃料成本（应用时间折现和单位转换）
            if ship.fuel_costs:
                for quarter_idx, fuel_cost in enumerate(ship.fuel_costs):
                    quarter = quarter_idx + 1  # 假设从第1季度开始
                    total_fuel_costs += fuel_cost * UNIT_SCALE * get_discount_factor(quarter)
            
            # 改造成本（应用时间折现和单位转换）
            if hasattr(ship, 'retrofit_start_quarter') and ship.retrofit_start_quarter:
                total_retrofit_costs += ship.retrofit_capex * UNIT_SCALE * get_discount_factor(ship.retrofit_start_quarter)
            else:
                total_retrofit_costs += ship.retrofit_capex * UNIT_SCALE * get_discount_factor(env.current_quarter)
            
            # 机会成本（应用时间折现和单位转换）
            if ship.opportunity_costs:
                for quarter_idx, opportunity_cost in enumerate(ship.opportunity_costs):
                    quarter = quarter_idx + 1  # 假设从第1季度开始
                    total_opportunity_costs += opportunity_cost * UNIT_SCALE * get_discount_factor(quarter)
            
            # 合规成本（应用时间折现和单位转换）
            for year, compliance_cost in ship.compliance_costs.items():
                year_end_quarter = (year - quarter_to_year(1) + 1) * 4  # 计算年末季度
                total_compliance_costs += compliance_cost * UNIT_SCALE * get_discount_factor(year_end_quarter)
            
            # SU收益（应用时间折现和单位转换）
            for year, su_revenue in ship.su_sale_revenues.items():
                year_end_quarter = (year - quarter_to_year(1) + 1) * 4  # 计算年末季度
                total_su_revenues += su_revenue * UNIT_SCALE * get_discount_factor(year_end_quarter)
            
            # ZNZ奖励（应用时间折现和单位转换）
            for year, znz_reward in ship.znz_rewards.items():
                year_end_quarter = (year - quarter_to_year(1) + 1) * 4  # 计算年末季度
                total_znz_rewards += znz_reward * UNIT_SCALE * get_discount_factor(year_end_quarter)
        
        # 计算总成本和净成本
        total_costs = total_fuel_costs + total_retrofit_costs + total_opportunity_costs + total_compliance_costs
        total_revenues = total_su_revenues + total_znz_rewards
        net_costs = total_costs - total_revenues
        
        # 总奖励应该等于净成本的负数（都应用了时间折现）
        total_rewards = -net_costs
        average_reward = total_rewards / len(episode_rewards) if episode_rewards else 0.0
        
        # 记录训练历史 - 修复奖励记录问题
        processed_total_rewards = sum(episode_rewards.values())  # 使用处理后的奖励
        training_history['iterations'].append(iteration + 1)
        training_history['total_emissions'].append(total_emissions)
        training_history['total_costs'].append(total_costs)
        training_history['total_rewards'].append(processed_total_rewards)  # 使用处理后的奖励
        training_history['average_rewards'].append(processed_total_rewards / len(episode_rewards) if episode_rewards else 0.0)
        training_history['ship_strategies'].append({
            'iteration': iteration + 1,
            'episode_strategies': episode_strategies
        })  # 新增：保存船舶策略信息
        
        if update_stats:
            training_history['actor_losses'].append(update_stats.get('actor_loss', 0.0))
            training_history['critic_losses'].append(update_stats.get('critic_loss', 0.0))
            training_history['entropies'].append(update_stats.get('entropy', 0.0))
            training_history['explained_variances'].append(update_stats.get('explained_variance', 0.0))
        else:
            training_history['actor_losses'].append(0.0)
            training_history['critic_losses'].append(0.0)
            training_history['entropies'].append(0.0)
            training_history['explained_variances'].append(0.0)
        
        # 打印进度 - 优化后的简洁格式
        iteration_time = time.time() - iteration_start_time
        if (iteration + 1) % 10 == 0 or iteration == 0:
            processed_total_rewards = sum(episode_rewards.values())
            
            # 修复：使用相同的数据源计算平均奖励
            # 存储当前迭代的环境奖励用于平均值计算
            if not hasattr(training_history, 'episode_rewards_history'):
                training_history['episode_rewards_history'] = []
            training_history['episode_rewards_history'].append(processed_total_rewards)
            
            # 计算最近10次迭代的平均奖励（使用环境奖励）
            recent_episode_rewards = training_history['episode_rewards_history'][-10:]
            avg_iteration_reward = sum(recent_episode_rewards) / len(recent_episode_rewards) if recent_episode_rewards else processed_total_rewards
            
            # 船队信息
            f_ships = sum(1 for ship in env.fleet.ships if ship.ship_type.value == 'F')
            g_ships = sum(1 for ship in env.fleet.ships if ship.ship_type.value in ['G', 'FG'])
            z_ships = sum(1 for ship in env.fleet.ships if ship.ship_type.value == 'Z_DF')
            
            print(f"[{iteration + 1:4d}/{num_iterations}] 船队:F{f_ships}/G{g_ships}/Z{z_ships} | "
                  f"排放:{total_emissions:7.1f} | 成本:{total_costs:10.1f} | "
                  f"奖励:{processed_total_rewards:7.3f} | 平均:{avg_iteration_reward:7.3f} | {iteration_time:.1f}s")
            
            # 每10次迭代显示详细成本和收益分解
            print(f"    成本分解: 燃料{total_fuel_costs:7.1f} | 改造{total_retrofit_costs:7.1f} | "
                  f"机会{total_opportunity_costs:7.1f} | 合规{total_compliance_costs:7.1f}")
            print(f"    收益分解: SU{total_su_revenues:7.1f} | ZNZ{total_znz_rewards:7.1f} | "
                  f"总收益{total_revenues:7.1f}")
        
        # 定期保存模型
        if (iteration + 1) % save_interval == 0:
            # 使用自定义保存目录或默认逻辑
            if save_dir:
                # 使用用户指定的保存目录
                os.makedirs(save_dir, exist_ok=True)
                model_path = f"{save_dir}/{algorithm_normalized}_{start_iteration + iteration + 1}.pth"
            else:
                # 为不同数据源创建独立的保存路径
                if 'MSC' in ships_csv_path:
                    default_save_dir = f"training_results/{algorithm_normalized}-msc"
                    os.makedirs(default_save_dir, exist_ok=True)
                    model_path = f"{default_save_dir}/{algorithm_normalized}-msc_{start_iteration + iteration + 1}.pth"
                elif 'Cosco' in ships_csv_path:
                    default_save_dir = f"training_results/{algorithm_normalized}-cosco"
                    os.makedirs(default_save_dir, exist_ok=True)
                    model_path = f"{default_save_dir}/{algorithm_normalized}-cosco_{start_iteration + iteration + 1}.pth"
                else:
                    model_path = f"{algorithm_normalized}_{start_iteration + iteration + 1}.pth"
            
            if hasattr(agent, 'save_models'):
                agent.save_models(model_path, ship_strategies=training_history['ship_strategies'])
            elif hasattr(agent, 'save_model'):
                agent.save_model(model_path, ship_strategies=training_history['ship_strategies'])
            elif hasattr(agent, 'save'):
                agent.save(model_path, ship_strategies=training_history['ship_strategies'])
        
        # 检查是否是当前最佳奖励，如果是则额外保存最佳模型
        current_total_reward = sum(episode_rewards.values()) if episode_rewards else 0.0
        if not hasattr(train, '_best_reward') or current_total_reward > train._best_reward:
            train._best_reward = current_total_reward
            train._best_iteration = start_iteration + iteration + 1
            
            # 保存最佳模型
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                best_model_path = f"{save_dir}/{algorithm_normalized}_best.pth"
            else:
                if 'MSC' in ships_csv_path:
                    default_save_dir = f"training_results/{algorithm_normalized}-msc"
                    os.makedirs(default_save_dir, exist_ok=True)
                    best_model_path = f"{default_save_dir}/{algorithm_normalized}-msc_best.pth"
                elif 'Cosco' in ships_csv_path:
                    default_save_dir = f"training_results/{algorithm_normalized}-cosco"
                    os.makedirs(default_save_dir, exist_ok=True)
                    best_model_path = f"{default_save_dir}/{algorithm_normalized}-cosco_best.pth"
                else:
                    best_model_path = f"{algorithm_normalized}_best.pth"
            
            if hasattr(agent, 'save_models'):
                agent.save_models(best_model_path, ship_strategies=training_history['ship_strategies'])
            elif hasattr(agent, 'save_model'):
                agent.save_model(best_model_path, ship_strategies=training_history['ship_strategies'])
            elif hasattr(agent, 'save'):
                agent.save(best_model_path, ship_strategies=training_history['ship_strategies'])
            
            print(f"    保存最佳模型: 奖励 {current_total_reward:.2f} (第 {train._best_iteration} 次迭代)")
    
    # 训练完成 - 简化输出
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间:{total_time:.1f}s | 平均:{total_time/num_iterations:.2f}s/iter")
    
    # 显示最终训练统计信息
    if training_history['total_rewards']:
        final_total_reward = training_history['total_rewards'][-1]
        final_emissions = training_history['total_emissions'][-1]
        final_total_costs = training_history['total_costs'][-1]
        
        # 从奖励处理器获取缩放因子（添加安全检查）
        reward_stats = reward_processor.get_stats()
        reward_scale_factor = reward_stats.get('scale_factor', 1.0)  # 默认值为1.0
        
        # 计算处理后的奖励（应用缩放因子）
        final_processed_reward = final_total_reward * reward_scale_factor
        
        # 计算净成本（总成本减去收益，奖励是净成本的负数）
        final_net_cost = -final_total_reward
        
    # 显示最终GPU内存使用情况
    if device == 'cuda':
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 保存最终模型
    if save_dir:
        # 使用用户指定的保存目录
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = f"{save_dir}/{algorithm_normalized}_{start_iteration + num_iterations}.pth"
        history_path = f"{save_dir}/{algorithm_normalized}_training_history.json"
        plot_name = algorithm_normalized
    else:
        # 使用默认逻辑
        if 'MSC' in ships_csv_path:
            default_save_dir = f"training_results/{algorithm_normalized}-msc"
            os.makedirs(default_save_dir, exist_ok=True)
            final_model_path = f"{default_save_dir}/{algorithm_normalized}-msc_{start_iteration + num_iterations}.pth"
            history_path = f"{default_save_dir}/{algorithm_normalized}-msc_training_history.json"
            plot_name = f"{algorithm_normalized}-msc"
        elif 'Cosco' in ships_csv_path:
            default_save_dir = f"training_results/{algorithm_normalized}-cosco"
            os.makedirs(default_save_dir, exist_ok=True)
            final_model_path = f"{default_save_dir}/{algorithm_normalized}-cosco_{start_iteration + num_iterations}.pth"
            history_path = f"{default_save_dir}/{algorithm_normalized}-cosco_training_history.json"
            plot_name = f"{algorithm_normalized}-cosco"
        else:
            final_model_path = f"{algorithm_normalized}_{start_iteration + num_iterations}.pth"
            history_path = f"{algorithm_normalized}_training_history.json"
            plot_name = algorithm_normalized
    
    if hasattr(agent, 'save_models'):
        agent.save_models(final_model_path)
    elif hasattr(agent, 'save_model'):
        agent.save_model(final_model_path)
    elif hasattr(agent, 'save'):
        agent.save(final_model_path)
    
    # 保存训练历史
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(training_history, plot_name, save_dir)
    
    # 绘制归一化指标
    plot_normalized_metrics(training_history, plot_name, reward_processor, save_dir)
    
    return agent, training_history

def evaluate(algorithm: str = 'mappo',
            model_path: str = None,
            num_episodes: int = 10,
            num_ships: int = 100,
            random_seed: int = 42,
            device: str = 'cpu',
            use_real_ships: bool = True,
            ships_csv_path: str = 'Container_ship.xlsx'):
    """评估训练好的模型 - 现在必须使用真实船舶数据"""
    
    print(f"开始{algorithm.upper()}评估 ({num_episodes}轮)")
    print(f"模型:{model_path} | 船舶:{num_ships}艘 | 种子:{random_seed}")
    
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 创建环境 - 必须使用真实船舶数据
    if use_real_ships and ships_csv_path:
        if ships_csv_path.endswith('.xlsx'):
            from load_real_ships import load_ships_from_excel
            # 传递None以使用所有船舶
            real_fleet = load_ships_from_excel(ships_csv_path, None)
        else:
            real_fleet = load_ships_from_csv(ships_csv_path, num_ships)
        
        env = MaritimeDecarbonizationEnv(
            num_ships=len(real_fleet.ships), 
            random_seed=random_seed, 
            fleet=real_fleet,
            bio_supply_cap_per_quarter=args.bio_cap_gj,
            bio_supply_ratio=args.bio_cap_ratio
        )
    else:
        raise ValueError("评估必须使用真实船舶数据，请设置 use_real_ships=True 和 ships_csv_path")
    
    # 获取观察和动作空间维度
    obs_dim = 36  # 单个智能体观察维度
    global_obs_dim = 54 + num_ships * 23  # 全局观察维度
    
    # 创建动作空间字典（每艘船作为一个智能体）
    action_dims = {}
    for ship in env.fleet.ships:
        if ship.ship_type.value == 'F':
            action_dims[ship.ship_id] = 10  # F类船舶：10个动作（包括改造）
        elif ship.ship_type.value in ['G', 'FG']:
            action_dims[ship.ship_id] = 5   # G/FG类船舶：5个动作
        elif ship.ship_type.value == 'Z_DF':
            action_dims[ship.ship_id] = 1   # Z-DF类船舶：1个动作
    
    # 根据选择的算法创建智能体
    if algorithm.lower() == 'mappo':
        agent = MAPPOAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            device=device
        )
    elif algorithm.lower() == 'gnn-mappo':
        from gnn_mappo_optimized import OptimizedGNNMAPPOAgent
        agent = OptimizedGNNMAPPOAgent(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dims=action_dims,
            max_episode_length=50,
            device=device,
            enable_gnn=True,
            hidden_dim=64
        )
    elif algorithm.lower() == 'qmix':
        agent = QMIXAgent(
            fleet=env.fleet,
            obs_dim=obs_dim,
            global_state_dim=global_obs_dim,
            device=device
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}，目前支持: 'mappo', 'gnn-mappo', 'qmix'")
    
    # 加载模型
    if model_path:
        agent.load_models(model_path)
    
    # 评估结果
    eval_results = {
        'total_emissions': [],
        'total_costs': [],
        'total_rewards': [],
        'episode_lengths': []
    }
    
    for episode in range(num_episodes):
        # 重置环境
        observations = env.reset()
        global_obs = create_global_observation(env)
        
        episode_rewards = {ship_id: 0.0 for ship_id in action_dims.keys()}
        episode_steps = 0
        
        # 运行一个episode
        max_episode_length = getattr(agent, 'max_episode_length', 50)
        while not env.done and episode_steps < max_episode_length:
            # 获取动作
            if algorithm.lower() == 'qmix':
                actions = agent.get_actions(observations, global_obs, training=False)
            else:
                actions, _ = agent.get_actions(observations, deterministic=True)
            
            # 执行动作
            next_observations, rewards, dones, info = env.step(actions)
            next_global_obs = create_global_observation(env)
            
            # 累计奖励（使用原始奖励进行统计）
            for ship_id, reward in rewards.items():
                episode_rewards[ship_id] += reward
            
            # 更新状态
            observations = next_observations
            global_obs = next_global_obs
            episode_steps += 1
        
        # 计算统计信息
        total_emissions = 0
        total_fuel_costs = 0
        total_retrofit_costs = 0
        total_opportunity_costs = 0
        total_compliance_costs = 0
        total_su_revenues = 0
        total_znz_rewards = 0
        
        # 统一单位转换因子：USD -> 百万USD（与环境保持一致）
        UNIT_SCALE = 1e-6  # 1百万美元 = 1,000,000美元
        
        for ship in env.fleet.ships:
            # 排放统计
            if ship.quarterly_emissions:
                total_emissions += ship.quarterly_emissions[-1]
            
            # 燃料成本（应用统一单位转换因子）
            if ship.fuel_costs:
                total_fuel_costs += sum(ship.fuel_costs) * UNIT_SCALE
            
            # 改造成本（应用统一单位转换因子）
            total_retrofit_costs += ship.retrofit_capex * UNIT_SCALE
            
            # 机会成本（应用统一单位转换因子）
            if ship.opportunity_costs:
                total_opportunity_costs += sum(ship.opportunity_costs) * UNIT_SCALE
            
            # 合规成本（应用统一单位转换因子）
            total_compliance_costs += sum(ship.compliance_costs.values()) * UNIT_SCALE
            
            # SU收益（应用统一单位转换因子）
            total_su_revenues += sum(ship.su_sale_revenues.values()) * UNIT_SCALE
            
            # ZNZ奖励（应用统一单位转换因子）
            total_znz_rewards += sum(ship.znz_rewards.values()) * UNIT_SCALE
        
        # 计算总成本和净成本
        total_costs = total_fuel_costs + total_retrofit_costs + total_opportunity_costs + total_compliance_costs
        total_revenues = total_su_revenues + total_znz_rewards
        net_costs = total_costs - total_revenues
        
        total_rewards = sum(episode_rewards.values())
        
        # 记录评估结果
        eval_results['total_emissions'].append(total_emissions)
        eval_results['total_costs'].append(total_costs)
        eval_results['total_rewards'].append(total_rewards)
        eval_results['episode_lengths'].append(episode_steps)
        
        print(f"评估轮次 {episode + 1:2d}/{num_episodes} | "
              f"总排放: {total_emissions:8.2f} | "
              f"净成本: {net_costs:10.2f} | "
              f"总奖励: {total_rewards:8.4f} | "
              f"步数: {episode_steps}")
        print(f"         成本分解 - 燃料: {total_fuel_costs:8.2f} | "
              f"改造: {total_retrofit_costs:8.2f} | "
              f"机会: {total_opportunity_costs:8.2f} | "
              f"合规: {total_compliance_costs:8.2f}")
        print(f"         收益分解 - SU收益: {total_su_revenues:8.2f} | "
              f"ZNZ奖励: {total_znz_rewards:8.2f} | "
              f"总成本: {total_costs:8.2f} | "
              f"总收益: {total_revenues:8.2f}")
    
    # 计算平均结果
    avg_emissions = np.mean(eval_results['total_emissions'])
    avg_costs = np.mean(eval_results['total_costs'])
    avg_rewards = np.mean(eval_results['total_rewards'])
    avg_steps = np.mean(eval_results['episode_lengths'])
    
    print(f"\n评估完成! 排放:{avg_emissions:.1f} | 成本:{avg_costs:.1f} | 奖励:{avg_rewards:.2f} | 步数:{avg_steps:.1f}")
    
    # 保存评估结果
    eval_path = f"{algorithm}_eval_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return eval_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='集成式海运脱碳训练')
    parser.add_argument('--algorithm', type=str, default='mappo', choices=['mappo', 'gnn-mappo', 'gru-mappo', 'gru-mappo-communication', 'gru-mappo-gnn', 'qmix'],
                        help='训练算法 (mappo, gnn-mappo, gru-mappo, gru-mappo-communication, gru-mappo-gnn)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='运行模式 (train, evaluate)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='训练迭代次数')
    parser.add_argument('--num_ships', type=int, default=100,
                        help='船舶数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='模型保存间隔')
    parser.add_argument('--model_path', type=str, default=None,
                        help='评估模式下的模型路径')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估模式下的评估轮数')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备 (auto, cpu, cuda)')
    parser.add_argument('--use_adaptive_scheduler', action='store_true', default=True,
                        help='是否使用自适应参数调度器')
    parser.add_argument('--scheduler_strategy', type=str, default='hybrid',
                        choices=['performance', 'time', 'hybrid'],
                        help='调度器策略 (performance, time, hybrid)')
    parser.add_argument('--use_real_ships', action='store_true', default=True,
                        help='是否使用真实船舶数据 (现在默认为True)')
    parser.add_argument('--ships_csv_path', type=str, default='Container_ship.xlsx',
                        help='真实船舶数据文件路径 (支持CSV和Excel格式)')
    parser.add_argument('--bio_cap_gj', type=float, default=None,
                        help='每季度可用的生物柴油供给上限 (GJ)，为空则不限制')
    parser.add_argument('--bio_cap_ratio', type=float, default=None,
                        help='固定比例上限 (0-1)，与基准GJ相乘得到每季度供给上限')
    parser.add_argument('--bio_cap_base_gj', type=float, default=None,
                        help='供给上限的固定基准GJ；不提供则自动取全年四季度峰值能耗')
    parser.add_argument('--load_model', type=str, default=None,
                        help='加载已保存的模型路径以继续训练')
    parser.add_argument('--start_iteration', type=int, default=0,
                        help='起始迭代序号，用于继续训练时的模型保存序号')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='自定义保存目录路径')
    
    args = parser.parse_args()
    
    # 设备检查
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            args.device = 'cpu'
    
    # 根据模式运行
    if args.mode == 'train':
        train(
            algorithm=args.algorithm,
            num_iterations=args.iterations,
            num_ships=args.num_ships,
            random_seed=args.seed,
            save_interval=args.save_interval,
            device=args.device,
            use_adaptive_scheduler=args.use_adaptive_scheduler,
            scheduler_strategy=args.scheduler_strategy,
            use_real_ships=args.use_real_ships,
            ships_csv_path=args.ships_csv_path,
            load_model=args.load_model,
            start_iteration=args.start_iteration,
            save_dir=args.save_dir,
            bio_cap_gj=args.bio_cap_gj,
            bio_cap_ratio=args.bio_cap_ratio,
            bio_cap_base_gj=args.bio_cap_base_gj
        )
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("错误: 评估模式需要指定模型路径 (--model_path)")
            return
        
        evaluate(
            algorithm=args.algorithm,
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            num_ships=args.num_ships,
            random_seed=args.seed,
            device=args.device
        )

if __name__ == "__main__":
    main()
