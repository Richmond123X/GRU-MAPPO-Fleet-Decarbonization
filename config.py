"""配置文件 - 海运脱碳系统参数"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class SystemConfig:
    """系统基础配置"""
    # 时间参数
    START_YEAR = 2026
    END_YEAR = 2035
    START_QUARTER = 1  # 2026Q1
    END_QUARTER = 40   # 2035Q4
    QUARTERS_PER_YEAR = 4
    DAYS_PER_QUARTER = 91.25
    
    # 政策参数
    POLICY_BASE_YEAR = 2008
    BASE_EMISSION_INTENSITY = 95.0  # gCO₂e/MJ
    SHADOW_PERIOD_START = 2026
    SHADOW_PERIOD_END = 2027
    POLICY_EFFECTIVE_START = 2028
    
    # 经济参数
    discount_rate = 0.06  # 6%年化贴现率
    QUARTERLY_DISCOUNT_RATE = (1 + discount_rate) ** (1/4) - 1  # 1.457%
    INFLATION_RATE = 0.025  # 2.5%年通胀率
    TECH_IMPROVEMENT_RATE = 0.02  # 2%年技术改进率
    
    # 船舶参数
    MAX_SERVICE_LIFE = 120  # 30年 = 120季度
    ENERGY_GROWTH_RATE = -0.007  # -0.7%年能耗增长率
    
    # SU管理参数
    SU_VALIDITY_PERIOD = 8  # 8季度有效期
    

    
    # 政策价格
    # 碳单位价格 (USD/tCO₂e)
    RU1_PRICE = 100  # USD/tCO₂e
    RU2_PRICE = 380  # USD/tCO₂e
    SU_SALE_PRICE = 100  # USD/tCO₂e
    ZNZ_REWARD_PRICE = 100  # USD/tCO₂e

@dataclass
class TEUConfig:
    """TEU级别配置"""
    # TEU级别定义
    # 注意：功率(kW)现在根据TEU容量动态计算，公式为：功率 = 4 * 容量(TEU)
    TEU_CLASSES = {
        'C1': {'range': (0, 999), 'retrofit_duration': 1, 'daily_charter': 8000},
        'C2': {'range': (1000, 1999), 'retrofit_duration': 1, 'daily_charter': 12000},
        'C3': {'range': (2000, 2999), 'retrofit_duration': 1, 'daily_charter': 16000},
        'C4': {'range': (3000, 4999), 'retrofit_duration': 1, 'daily_charter': 24000},
        'C5': {'range': (5000, 7999), 'retrofit_duration': 1, 'daily_charter': 32000},
        'C6': {'range': (8000, 11999), 'retrofit_duration': 1, 'daily_charter': 40000},
        'C7': {'range': (12000, 14499), 'retrofit_duration': 2, 'daily_charter': 48000},
        'C8': {'range': (14500, 19999), 'retrofit_duration': 2, 'daily_charter': 56000},
        'C9': {'range': (20000, float('inf')), 'retrofit_duration': 2, 'daily_charter': 64000}
    }
    
    # 季度基准能耗 (GJ) - 各TEU级别船舶的季度能耗基准
    QUARTERLY_ENERGY_BASELINE = {
        'C1': [28016, 25441, 28967, 29728],
        'C2': [64429, 51629, 63111, 67370],
        'C3': [91473, 69392, 88443, 96514],
        'C4': [147853, 116703, 150522, 153923],
        'C5': [206749, 167490, 205235, 213539],
        'C6': [266501, 215861, 266512, 273037],
        'C7': [279149, 231600, 281005, 291009],
        'C8': [265605, 221105, 256124, 268798],
        'C9': [362910, 288732, 359076, 362126],
    }

@dataclass
class FuelConfig:
    """燃料配置"""
    # 基础燃料参数
    # emission_factor: g/MJ, price: USD/GJ, energy_density: MJ/kg
    BASE_FUELS = {
        'F': {'name': 'VLSFO', 'emission_factor': 95.0, 'price': 14.0, 'energy_density': 40.4},
        'G': {'name': 'LNG', 'emission_factor': 80.0, 'price': 13.8, 'energy_density': 50.0},
        'B': {'name': 'HVO', 'emission_factor': 30.0, 'price': 29.5, 'energy_density': 44.0},
        'Z': {'name': 'Green Methanol', 'emission_factor': 19.0, 'price': 26.8, 'energy_density': 19.9},
        'BG': {'name': 'Bio-LNG', 'emission_factor': 30.0, 'price': 20.0, 'energy_density': 50.0}
    }
    
    # F类船舶混合燃料策略
    F_CLASS_STRATEGIES = {
        'B0': {'F': 1.0, 'B': 0.0},
        'B5': {'F': 0.95, 'B': 0.05},
        'B7': {'F': 0.93, 'B': 0.07},
        'B10': {'F': 0.90, 'B': 0.10},
        'B20': {'F': 0.80, 'B': 0.20},
        'B30': {'F': 0.70, 'B': 0.30},
        'B50': {'F': 0.50, 'B': 0.50},
        'B100': {'F': 0.0, 'B': 1.0}
    }
    
    # G类船舶混合燃料策略
    G_CLASS_STRATEGIES = {
        'G0': {'G': 1.0, 'BG': 0.0},
        'G5': {'G': 0.95, 'BG': 0.05},
        'G10': {'G': 0.90, 'BG': 0.10},
        'G20': {'G': 0.80, 'BG': 0.20},
        'G50': {'G': 0.50, 'BG': 0.50}
    }
    
    # Z-DF类船舶策略
    Z_CLASS_STRATEGIES = {
        'Z100': {'Z': 1.0}
    }

@dataclass
class PolicyConfig:
    """政策目标配置"""
    # 政策严格程度倍数
    strictness = 1.0
    
    # 年度减排目标 (相对2008年基准)
    BASE_TARGETS = {
        2028: 0.04, 2029: 0.08, 2030: 0.14, 2031: 0.20,
        2032: 0.23, 2033: 0.26, 2034: 0.28, 2035: 0.30
    }
    
    DIRECT_TARGETS = {
        2028: 0.17, 2029: 0.21, 2030: 0.27, 2031: 0.33,
        2032: 0.36, 2033: 0.39, 2034: 0.41, 2035: 0.43
    }
    
    @staticmethod
    def get_base_intensity(year: int) -> float:
        """获取Base目标强度"""
        if year < 2028:
            return SystemConfig.BASE_EMISSION_INTENSITY
        target = PolicyConfig.BASE_TARGETS.get(year, 0.30)
        return SystemConfig.BASE_EMISSION_INTENSITY * (1 - target)
    
    @staticmethod
    def get_direct_intensity(year: int) -> float:
        """获取Direct目标强度"""
        if year < 2028:
            return SystemConfig.BASE_EMISSION_INTENSITY
        target = PolicyConfig.DIRECT_TARGETS.get(year, 0.43)
        return SystemConfig.BASE_EMISSION_INTENSITY * (1 - target)

@dataclass
class RetrofitConfig:
    """改造配置"""
    BASE_PRICE = 392  # USD/kW (2015年基准价格)
    BASE_YEAR = 2015
    
    @staticmethod
    def calculate_retrofit_cost(engine_power: float, year: int) -> float:
        """计算改造成本"""
        inflation_factor = (1 + SystemConfig.INFLATION_RATE) ** (year - RetrofitConfig.BASE_YEAR)
        tech_factor = max(0.7, 1 - SystemConfig.TECH_IMPROVEMENT_RATE * (year - RetrofitConfig.BASE_YEAR))
        adjusted_price = RetrofitConfig.BASE_PRICE * inflation_factor * tech_factor
        return engine_power * adjusted_price

@dataclass
class TrainingConfig:
    """训练配置"""
    # PPO超参数 - 调整为更稳定的设置
    ACTOR_LR = 1e-4  # 降低Actor学习率
    CRITIC_LR = 5e-4  # 降低Critic学习率
    PPO_EPOCHS = 8  # 减少更新轮数
    CLIP_RATIO = 0.1  # 降低裁剪比例
    BATCH_SIZE = 64
    BUFFER_SIZE = 2048
    GAE_LAMBDA = 0.95
    GAMMA = 0.99
    
    # 学习率调度策略配置
    LR_SCHEDULER_TYPE = "none"  # "cosine", "exponential", "step", "plateau", "none"
    
    # 指数衰减调度器参数
    LR_DECAY_GAMMA = 0.995  # 每次更新的衰减因子
    LR_DECAY_STEP = 50  # 每多少次迭代衰减一次
    
    # 余弦退火调度器参数
    COSINE_T_MAX = 5000  # 增大T_max，减缓衰减速度
    COSINE_ETA_MIN_RATIO = 0.1  # 最小学习率比例（相对于初始学习率）
    
    # 阶梯衰减调度器参数
    STEP_SIZE = 200  # 每200次迭代衰减
    STEP_GAMMA = 0.8  # 衰减因子
    
    # 平台调度器参数
    PLATEAU_PATIENCE = 100  # 等待改善的迭代次数
    PLATEAU_FACTOR = 0.5  # 衰减因子
    PLATEAU_THRESHOLD = 1e-4  # 改善阈值
    
    # 早停机制参数
    EARLY_STOPPING = False  # 禁用早停
    EARLY_STOPPING_PATIENCE = 200  # 早停耐心值
    EARLY_STOPPING_MIN_DELTA = 1.0  # 最小改善幅度
    EARLY_STOPPING_RESTORE_BEST = True  # 恢复最佳权重
    
    # 网络超参数
    ACTOR_GRU_HIDDEN = 64
    ACTOR_GRU_SEQUENCE = 8
    ACTOR_DROPOUT = 0.1
    
    CRITIC_GRU_HIDDEN = 128
    CRITIC_GRU_SEQUENCE = 12
    CRITIC_DROPOUT = 0.2
    
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 32
    
    # 梯度裁剪
    ACTOR_GRAD_CLIP = 0.5
    CRITIC_GRAD_CLIP = 1.0
    
    # 权重衰减
    ACTOR_WEIGHT_DECAY = 1e-5
    CRITIC_WEIGHT_DECAY = 5e-4
    
    # 训练参数
    MAX_EPISODES = 1000
    EVAL_FREQUENCY = 50
    SAVE_FREQUENCY = 100
    
    # 随机种子
    RANDOM_SEED = 42

# 全局配置实例
sys_config = SystemConfig()
teu_config = TEUConfig()
fuel_config = FuelConfig()
policy_config = PolicyConfig()
retrofit_config = RetrofitConfig()
training_config = TrainingConfig()

# 辅助函数
def get_discount_factor(quarter: int) -> float:
    """获取季度贴现因子"""
    return 1 / (1 + sys_config.QUARTERLY_DISCOUNT_RATE) ** quarter

def quarter_to_year(quarter: int) -> int:
    """季度转年份"""
    return sys_config.START_YEAR + (quarter - 1) // sys_config.QUARTERS_PER_YEAR

def get_quarter_in_year(quarter: int) -> int:
    """获取年内季度 (1-4)"""
    return ((quarter - 1) % sys_config.QUARTERS_PER_YEAR) + 1

def calculate_mixed_fuel_properties(fuel_mix: Dict[str, float]) -> Tuple[float, float]:
    """计算混合燃料的排放因子和价格"""
    emission_factor = sum(ratio * fuel_config.BASE_FUELS[fuel]['emission_factor'] 
                         for fuel, ratio in fuel_mix.items())
    price = sum(ratio * fuel_config.BASE_FUELS[fuel]['price'] 
               for fuel, ratio in fuel_mix.items())
    return emission_factor, price