"""多智能体海运脱碳环境"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import copy

from ship import Fleet, Ship, ShipType, ShipState
from config import (
    sys_config, teu_config, fuel_config, policy_config,
    get_discount_factor, quarter_to_year, get_quarter_in_year
)

class MaritimeDecarbonizationEnv(gym.Env):
    """海运脱碳多智能体环境"""
    
    def __init__(self, num_ships: int = 100, random_seed: int = 42, fleet: Fleet = None, bio_supply_cap_per_quarter: float = None, bio_supply_ratio: float = None, bio_supply_base_gj: float = None):
        super().__init__()
        
        self.num_ships = num_ships
        self.random_seed = random_seed
        
        # 初始化船队 - 必须从外部传入真实船队
        if fleet is None:
            raise ValueError("必须提供真实船队数据，不再支持随机生成船队")
        
        self.fleet = fleet
        self.initial_fleet = copy.deepcopy(fleet)
        
        # 环境状态
        self.current_quarter = 1
        self.done = False
        
        # 历史记录
        self.episode_rewards = []
        self.episode_emissions = []
        self.episode_costs = []
        
        # 定义动作空间和观测空间
        self._setup_action_observation_spaces()
        
        # 性能指标
        self.total_pv_cost = 0.0
        self.quarterly_metrics = []
        self.bio_supply_cap_per_quarter = bio_supply_cap_per_quarter
        self.bio_supply_ratio = bio_supply_ratio
        self.bio_supply_base_gj = bio_supply_base_gj
        self._bio_supply_remaining = bio_supply_cap_per_quarter
        self._bio_consumed_this_quarter = 0.0
        if self.bio_supply_ratio is not None and self.bio_supply_base_gj is None and self.fleet is not None:
            self.bio_supply_base_gj = self._compute_peak_quarterly_energy()

    def set_biodiesel_supply_cap(self, cap_gj_per_quarter: float):
        self.bio_supply_cap_per_quarter = cap_gj_per_quarter
        self._bio_supply_remaining = cap_gj_per_quarter

    def set_biodiesel_supply_ratio(self, ratio: float):
        self.bio_supply_ratio = ratio
    
    def set_biodiesel_supply_base(self, base_gj: float):
        self.bio_supply_base_gj = base_gj
        
    def _setup_action_observation_spaces(self):
        """设置动作和观测空间"""
        self.action_spaces = {}
        
        # F类船舶：根据状态动态设置动作空间
        # OPERATING状态：9个动作（燃料策略 + START_RETROFIT）
        f_operating_actions = ['B0', 'B5', 'B7', 'B10', 'B20', 'B30', 'B50', 'B100', 'START_RETROFIT']
        # RETROFITTING状态：1个动作（RETROFIT_CONTINUE）
        f_retrofitting_actions = ['RETROFIT_CONTINUE']
        
        # G/FG类船舶：5个动作
        g_actions = ['G0', 'G5', 'G10', 'G20', 'G50']
        # Z-DF类船舶：1个动作
        z_actions = ['Z100']
        
        for ship in self.fleet.ships:
            if ship.ship_type == ShipType.F:
                if ship.state == ShipState.RETROFITTING:
                    self.action_spaces[ship.ship_id] = spaces.Discrete(len(f_retrofitting_actions))
                else:
                    self.action_spaces[ship.ship_id] = spaces.Discrete(len(f_operating_actions))
            elif ship.ship_type in [ShipType.G, ShipType.FG]:
                self.action_spaces[ship.ship_id] = spaces.Discrete(len(g_actions))
            elif ship.ship_type == ShipType.Z_DF:
                self.action_spaces[ship.ship_id] = spaces.Discrete(len(z_actions))
        
        # 观测空间：每个智能体的状态向量 (36维)
        # 船舶状态24维 + 时间编码12维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )
        
        # 全局状态空间 (用于Critic)
        # 全局状态42维 + 时间编码12维 + 船舶状态集合N×23维
        self.global_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(54 + self.num_ships * 23,), dtype=np.float32
        )
    
    def reset(self) -> Dict[int, np.ndarray]:
        """重置环境"""
        # 重置船队
        self.fleet = copy.deepcopy(self.initial_fleet)
        self.current_quarter = 1
        self.done = False
        
        # 重置历史记录
        self.episode_rewards = []
        self.episode_emissions = []
        self.episode_costs = []
        self.total_pv_cost = 0.0
        self.quarterly_metrics = []
        
        # 返回初始观测 - 优化版本
        observations = {}
        active_ships = self.fleet.get_active_ships()
        
        # 预计算时间编码，避免重复计算
        if not hasattr(self, '_cached_time_encoding') or self._cached_quarter != self.current_quarter:
            self._cached_time_encoding = self._compute_time_encoding()
            self._cached_quarter = self.current_quarter
        
        # 批量处理船舶观测
        for ship in active_ships:
            observations[ship.ship_id] = self._get_ship_observation_optimized(ship)
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """环境步进"""
        # 转换动作索引为动作字符串
        action_strings = self._convert_actions(actions)

        # 每季度重置可用生物柴油供给
        if self.bio_supply_ratio is not None:
            base = float(self.bio_supply_base_gj or 0.0)
            self._bio_supply_remaining = self.bio_supply_ratio * base
        elif self.bio_supply_cap_per_quarter is not None:
            self._bio_supply_remaining = self.bio_supply_cap_per_quarter
        self._bio_consumed_this_quarter = 0.0

        # 执行动作
        self._execute_actions(action_strings)
        
        # 计算奖励
        rewards = self._calculate_rewards()
        
        # 更新环境状态
        self.current_quarter += 1
        
        # 检查是否结束
        self.done = (self.current_quarter > sys_config.END_QUARTER)
        
        # 获取新的观测
        observations = {}
        dones = {}
        for ship in self.fleet.get_active_ships():
            observations[ship.ship_id] = self._get_ship_observation(ship)
            dones[ship.ship_id] = self.done or ship.state == ShipState.RETIRED
        
        # 记录指标
        self._record_metrics(rewards)
        
        # 信息字典
        info = {
            'quarter': self.current_quarter - 1,
            'year': quarter_to_year(self.current_quarter - 1),
            'total_emissions': sum(ship.quarterly_emissions[-1] if ship.quarterly_emissions else 0 
                                 for ship in self.fleet.get_active_ships()),
            'total_reward': sum(rewards.values()),
            'active_ships': len(self.fleet.get_active_ships()),
            'fleet_stats': self.fleet.get_fleet_statistics()
        }
        if self.bio_supply_cap_per_quarter is not None or self.bio_supply_ratio is not None:
            info['bio_supply_remaining'] = max(0.0, float(self._bio_supply_remaining or 0.0))
            info['bio_supply_consumed'] = float(self._bio_consumed_this_quarter)
            info['bio_supply_base_gj'] = float(self.bio_supply_base_gj or 0.0)
        
        return observations, rewards, dones, info
    
    def _convert_actions(self, actions: Dict[int, int]) -> Dict[int, str]:
        """转换动作索引为动作字符串"""
        action_strings = {}
        
        # 根据船舶状态动态定义动作映射
        f_operating_actions = ['B0', 'B5', 'B7', 'B10', 'B20', 'B30', 'B50', 'B100', 'START_RETROFIT']
        f_retrofitting_actions = ['RETROFIT_CONTINUE']
        g_actions = ['G0', 'G5', 'G10', 'G20', 'G50']
        z_actions = ['Z100']
        
        for ship_id, action_idx in actions.items():
            ship = next((s for s in self.fleet.ships if s.ship_id == ship_id), None)
            if ship is None or ship.state == ShipState.RETIRED:
                continue
            
            if ship.ship_type == ShipType.F:
                if ship.state == ShipState.RETROFITTING:
                    # 改装中的船舶只能选择RETROFIT_CONTINUE
                    if 0 <= action_idx < len(f_retrofitting_actions):
                        action_strings[ship_id] = f_retrofitting_actions[action_idx]
                    else:
                        action_strings[ship_id] = 'RETROFIT_CONTINUE'  # 默认动作
                else:
                    # 运营中的船舶可以选择燃料策略或开始改装
                    if 0 <= action_idx < len(f_operating_actions):
                        action_strings[ship_id] = f_operating_actions[action_idx]
                    else:
                        action_strings[ship_id] = 'B0'  # 默认动作
            elif ship.ship_type in [ShipType.G, ShipType.FG]:
                if 0 <= action_idx < len(g_actions):
                    action_strings[ship_id] = g_actions[action_idx]
                else:
                    action_strings[ship_id] = 'G0'  # 默认动作
            elif ship.ship_type == ShipType.Z_DF:
                action_strings[ship_id] = 'Z100'  # 唯一动作
        
        return action_strings
    
    def _execute_actions(self, actions: Dict[int, str]):
        """执行动作"""
        for ship in self.fleet.get_active_ships():
            action = actions.get(ship.ship_id, 'B0')

            # 供给约束：为F类船舶燃料策略动作在执行前应用生物柴油供给限制
            if self.bio_supply_cap_per_quarter is not None and ship.ship_type == ShipType.F:
                action = self._apply_bio_supply_constraint(ship, action)
            
            # 船龄增加
            ship.age_one_quarter()
            
            # 更新SU库存
            ship.update_su_inventory(self.current_quarter)
            
            # 硬性约束：改装逻辑验证和强制执行
            action = self._validate_and_enforce_retrofit_logic(ship, action)
            
            # 执行具体动作
            if action == 'START_RETROFIT':
                success = ship.start_retrofit(self.current_quarter)
                if not success:
                    # 如果改装失败，强制使用默认燃料策略
                    if ship.ship_type == ShipType.F:
                        action = 'B0'
                    elif ship.ship_type in [ShipType.G, ShipType.FG]:
                        action = 'G0'
                    elif ship.ship_type == ShipType.Z_DF:
                        action = 'Z100'
            elif action == 'RETROFIT_CONTINUE':
                retrofit_completed = ship.continue_retrofit()
                if retrofit_completed:
                    # 改装完成后，强制更新动作空间
                    self._update_ship_action_space_after_retrofit(ship)
            else:
                # 燃料策略动作
                if ship.state == ShipState.OPERATING:
                    # 硬性约束：Z_DF类船舶只能使用Z100
                    if ship.ship_type == ShipType.Z_DF and action != 'Z100':
                        action = 'Z100'
                    ship.update_quarterly_data(self.current_quarter, action)
            
            # 计算机会成本
            opportunity_cost = ship.calculate_opportunity_cost()
            ship.opportunity_costs.append(opportunity_cost)
        
        # 年末处理 (Q4)
        if get_quarter_in_year(self.current_quarter) == 4:
            self._process_year_end()

    def _apply_bio_supply_constraint(self, ship: 'Ship', action: str) -> str:
        from config import fuel_config
        # 仅对运营期燃料策略动作应用供给约束
        if action in fuel_config.F_CLASS_STRATEGIES:
            fuel_mix = fuel_config.F_CLASS_STRATEGIES[action]
            b_frac = float(fuel_mix.get('B', 0.0))
            energy = ship.calculate_quarterly_energy(self.current_quarter)
            demand = energy * b_frac
            if demand <= (self._bio_supply_remaining or 0.0):
                self._bio_supply_remaining -= demand
                self._bio_consumed_this_quarter += demand
                return action
            # 尝试降级到更低的生物柴油占比
            degrade_order = ['B50', 'B30', 'B20', 'B10', 'B7', 'B5', 'B0']
            for cand in degrade_order:
                fm = fuel_config.F_CLASS_STRATEGIES.get(cand)
                if not fm:
                    continue
                cand_b = float(fm.get('B', 0.0))
                cand_demand = energy * cand_b
                if cand_demand <= (self._bio_supply_remaining or 0.0):
                    self._bio_supply_remaining -= cand_demand
                    self._bio_consumed_this_quarter += cand_demand
                    return cand
            # 如果完全没有供给，退回B0
            return 'B0'
        return action

    def _compute_peak_quarterly_energy(self) -> float:
        peak = 0.0
        for q in range(1, 5):
            total = 0.0
            for ship in self.fleet.ships:
                total += ship.calculate_quarterly_energy(q)
            peak = max(peak, total)
        return peak
    
    def _validate_and_enforce_retrofit_logic(self, ship: 'Ship', action: str) -> str:
        """验证并强制执行改装逻辑的硬性约束"""
        from ship import ShipState, ShipType
        
        # 硬性约束1：只有F类船舶可以开始改装
        if action == 'START_RETROFIT':
            if ship.ship_type != ShipType.F or ship.state != ShipState.OPERATING:
                # 强制使用默认燃料策略
                return 'B0' if ship.ship_type == ShipType.F else ('G0' if ship.ship_type in [ShipType.G, ShipType.FG] else 'Z100')
            
            # 硬性约束2：检查改装时机约束
            from config import sys_config
            latest_start = sys_config.END_QUARTER - ship.retrofit_duration + 1
            if self.current_quarter > latest_start:
                # 改装时机已过，强制使用默认燃料策略
                return 'B0'
        
        # 硬性约束3：只有正在改装的船舶可以继续改装
        elif action == 'RETROFIT_CONTINUE':
            if ship.state != ShipState.RETROFITTING:
                # 如果不在改装状态，强制使用适当的燃料策略
                if ship.ship_type == ShipType.F:
                    return 'B0'
                elif ship.ship_type in [ShipType.G, ShipType.FG]:
                    return 'G0'
                elif ship.ship_type == ShipType.Z_DF:
                    return 'Z100'
            
            # 硬性约束4：检查改装时间是否超过规定持续时间
            if ship.retrofit_progress >= ship.retrofit_duration:
                # 改装已完成，不能继续RETROFIT_CONTINUE
                return 'Z100'  # 改装完成的船舶应该使用绿色甲醇
        
        # 硬性约束5：改装中的船舶只能执行RETROFIT_CONTINUE
        elif ship.state == ShipState.RETROFITTING:
            return 'RETROFIT_CONTINUE'
        
        # 硬性约束6：Z_DF类船舶只能使用Z100策略
        elif ship.ship_type == ShipType.Z_DF and action != 'Z100':
            return 'Z100'
        
        # 硬性约束7：退役船舶不能执行任何动作
        elif ship.state == ShipState.RETIRED:
            return 'B0'  # 默认动作，但实际上退役船舶不会被处理
        
        return action
    
    def _update_ship_action_space_after_retrofit(self, ship: 'Ship'):
        """改装完成后更新船舶的动作空间"""
        from ship import ShipType
        import gymnasium as gym
        
        # 更新动作空间：Z_DF类船舶只有1个动作
        if ship.ship_type == ShipType.Z_DF:
            self.action_spaces[ship.ship_id] = gym.spaces.Discrete(1)  # 只有Z100一个动作
    
    def _process_year_end(self):
        """年末处理：计算SU/RU、合规成本等"""
        current_year = quarter_to_year(self.current_quarter)
        
        for ship in self.fleet.get_active_ships():
            # 计算年度SU和RU
            su, ru1, ru2 = ship.calculate_annual_su_ru(current_year)
            
            # 添加SU到库存
            if su > 0:
                ship.add_su_to_inventory(su, self.current_quarter)
                ship.su_history[current_year] = su
            
            # 计算合规成本和收益
            if current_year >= sys_config.POLICY_EFFECTIVE_START:
                # 使用SU抵消RU
                total_ru = ru1 + ru2
                used_su, net_ru = ship.use_su_for_ru(total_ru)
                
                # 计算净RU1和RU2比例
                if total_ru > 0:
                    ru1_ratio = ru1 / total_ru
                    ru2_ratio = ru2 / total_ru
                    net_ru1 = net_ru * ru1_ratio
                    net_ru2 = net_ru * ru2_ratio
                else:
                    net_ru1 = net_ru2 = 0
                
                # 合规成本
                compliance_cost = (
                    net_ru1 * sys_config.RU1_PRICE + 
                    net_ru2 * sys_config.RU2_PRICE
                )
                ship.compliance_costs[current_year] = compliance_cost
                
                # SU出售收益
                remaining_su = ship.get_available_su()
                su_sale_revenue = remaining_su * sys_config.SU_SALE_PRICE
                ship.su_sale_revenues[current_year] = su_sale_revenue
                
                # ZNZ奖励 - 只有甲醇双燃料船才能获得ZNZ奖励
                if su > 0 and ship.ship_type == ShipType.Z_DF:  # 有盈余单位且为甲醇双燃料船才有ZNZ奖励
                    znz_reward = su * sys_config.ZNZ_REWARD_PRICE
                    ship.znz_rewards[current_year] = znz_reward
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """计算奖励（修复版本，确保成本导向）"""
        rewards = {}
        
        # 统一单位转换因子：USD -> 百万USD
        UNIT_SCALE = 1e-6  # 1百万美元 = 1,000,000美元
        
        # 修正的奖励配置 - 确保成本导向且数量级合理
        BASE_REWARD_SCALE = 1.0  # 调整基础奖励缩放，使奖励与成本数量级接近
        PROGRESS_BONUS_SCALE = 0.02  # 减少进步奖励缩放
        EFFICIENCY_BONUS_SCALE = 0.01  # 减少效率奖励缩放
        REVENUE_SCALE = 0.3  # 大幅减少收益权重，确保成本占主导
        
        for ship in self.fleet.get_active_ships():
            total_costs = 0.0
            total_revenues = 0.0
            
            # === 基础成本计算（负奖励）===
            # 季度燃料成本
            if ship.fuel_costs:
                fuel_cost = ship.fuel_costs[-1] * UNIT_SCALE
                total_costs += fuel_cost * get_discount_factor(self.current_quarter)
            
            # 改造CAPEX成本
            if (ship.retrofit_start_quarter == self.current_quarter and 
                hasattr(ship, 'retrofit_capex')):
                retrofit_cost = ship.retrofit_capex * UNIT_SCALE
                total_costs += retrofit_cost * get_discount_factor(self.current_quarter)
            
            # 机会成本
            if ship.opportunity_costs:
                opportunity_cost = ship.opportunity_costs[-1] * UNIT_SCALE
                total_costs += opportunity_cost * get_discount_factor(self.current_quarter)
            
            # === 年末收益计算（政策生效期）===
            if (get_quarter_in_year(self.current_quarter) == 4 and 
                quarter_to_year(self.current_quarter) >= sys_config.POLICY_EFFECTIVE_START):
                
                current_year = quarter_to_year(self.current_quarter)
                year_end_quarter = self.current_quarter
                discount_factor = get_discount_factor(year_end_quarter)
                
                # 合规成本（增加到总成本）
                compliance_cost = ship.compliance_costs.get(current_year, 0) * UNIT_SCALE
                total_costs += compliance_cost * discount_factor
                
                # SU出售收益（正收益，但权重降低）
                su_revenue = ship.su_sale_revenues.get(current_year, 0) * UNIT_SCALE
                total_revenues += su_revenue * discount_factor * REVENUE_SCALE
                
                # ZNZ奖励（正收益，但权重降低）
                znz_reward = ship.znz_rewards.get(current_year, 0) * UNIT_SCALE
                total_revenues += znz_reward * discount_factor * REVENUE_SCALE
            
            # === 小幅奖励塑形组件 ===
            shaping_bonus = 0.0
            
            # 1. 排放效率奖励（减少）
            if ship.quarterly_emissions and ship.fuel_costs:
                current_emissions = ship.quarterly_emissions[-1]
                current_fuel_cost = ship.fuel_costs[-1]
                
                if current_emissions > 0 and len(ship.quarterly_emissions) > 1:
                    prev_emissions = ship.quarterly_emissions[-2]
                    prev_fuel_cost = ship.fuel_costs[-2] if len(ship.fuel_costs) > 1 else current_fuel_cost
                    
                    if prev_emissions > 0:
                        prev_efficiency = prev_fuel_cost / prev_emissions
                        current_efficiency = current_fuel_cost / current_emissions
                        efficiency_improvement = prev_efficiency - current_efficiency
                        if efficiency_improvement > 0:
                            shaping_bonus += efficiency_improvement * EFFICIENCY_BONUS_SCALE
            
            # 2. 排放减少奖励（减少）
            if len(ship.quarterly_emissions) >= 2:
                emission_reduction = ship.quarterly_emissions[-2] - ship.quarterly_emissions[-1]
                if emission_reduction > 0:
                    shaping_bonus += emission_reduction * 5e-7  # 进一步减少
            
            # 3. 改造进度奖励（减少）
            if hasattr(ship, 'retrofit_progress') and ship.retrofit_progress > 0:
                shaping_bonus += ship.retrofit_progress * PROGRESS_BONUS_SCALE
            
            # === 计算最终奖励（确保成本导向）===
            # 基础奖励 = 收益 - 成本（通常为负）
            base_reward = (total_revenues - total_costs) * BASE_REWARD_SCALE
            
            # 添加小幅塑形奖励
            final_reward = base_reward + shaping_bonus
            
            # 确保奖励不会变成过大的正数（成本应该占主导）
            if final_reward > 0:
                final_reward = final_reward * 0.1  # 大幅减少正奖励
            
            # 动态裁剪范围
            quarter_progress = self.current_quarter / 80.0
            clip_range = 10.0 + 40.0 * (1 - quarter_progress)
            
            # 应用裁剪
            clipped_reward = np.clip(final_reward, -clip_range, clip_range)
            
            rewards[ship.ship_id] = clipped_reward
        
        return rewards
    
    def _get_ship_observation(self, ship: Ship) -> np.ndarray:
        """获取船舶观测向量 (36维)"""
        obs = np.zeros(36, dtype=np.float32)
        
        # 船舶状态向量 (23维)
        idx = 0
        
        # 基础属性 (5维)
        # TEU级别 (1维)
        teu_level = int(ship.teu_class[1])  # C1->1, C2->2, ..., C9->9
        obs[idx] = teu_level / 9.0  # 归一化到[1/9, 1]
        idx += 1
        
        # 船舶类型 (4维) One-hot编码
        type_mapping = {'F': 0, 'G': 1, 'FG': 2, 'Z_DF': 3}
        type_idx = type_mapping.get(ship.ship_type.value, 0)
        obs[idx + type_idx] = 1.0
        idx += 4
        
        # 船队模块状态 (8维)
        # 船龄 (1维)
        obs[idx] = ship.age_quarters / sys_config.MAX_SERVICE_LIFE
        idx += 1
        
        # 改造状态 (3维) One-hot编码
        state_mapping = {'operating': 0, 'retrofitting': 1, 'retired': 2}
        state_idx = state_mapping.get(ship.state.value, 0)
        if state_idx < 3:
            obs[idx + state_idx] = 1.0
        idx += 3
        
        # 季度能耗 (1维)
        if ship.quarterly_energy:
            energy = ship.quarterly_energy[-1]
            obs[idx] = np.log(1 + energy) / 20.0  # 对数归一化
        idx += 1
        
        # 季度排放量 (1维)
        if ship.quarterly_emissions:
            emissions = ship.quarterly_emissions[-1]
            obs[idx] = np.log(1 + emissions) / 15.0  # 对数归一化
        idx += 1
        
        # 年度能耗累计 (1维)
        current_year = quarter_to_year(self.current_quarter)
        annual_energy = ship.annual_energy.get(current_year, 0)
        quarter_in_year = get_quarter_in_year(self.current_quarter)
        if quarter_in_year > 0 and annual_energy > 0:
            expected_annual = annual_energy / quarter_in_year * 4
            obs[idx] = annual_energy / expected_annual
        else:
            obs[idx] = 0.0
        idx += 1
        
        # 年度排放累计 (1维)
        annual_emissions = ship.annual_emissions.get(current_year, 0)
        if quarter_in_year > 0 and annual_emissions > 0:
            expected_annual_emissions = annual_emissions / quarter_in_year * 4
            obs[idx] = annual_emissions / expected_annual_emissions
        else:
            obs[idx] = 0.0
        idx += 1
        
        # 年度排放强度 (1维)
        annual_intensity = ship.annual_emission_intensity.get(current_year, 0)
        obs[idx] = annual_intensity / 100.0  # 归一化到[0,1]
        idx += 1
        
        # SU管理状态 (5维)
        # 当前SU库存总量 (1维)
        total_su = ship.get_available_su()
        obs[idx] = np.log(1 + total_su) / 10.0
        idx += 1
        
        # SU库存按到期时间分布 (4维)
        su_distribution = [0, 0, 0, 0]  # [1-2季度, 3-4季度, 5-6季度, 7-8季度]
        for su_entry in ship.su_inventory:
            quarters_to_expiry = su_entry.expiry_quarter - self.current_quarter
            if 1 <= quarters_to_expiry <= 2:
                su_distribution[0] += su_entry.amount
            elif 3 <= quarters_to_expiry <= 4:
                su_distribution[1] += su_entry.amount
            elif 5 <= quarters_to_expiry <= 6:
                su_distribution[2] += su_entry.amount
            elif 7 <= quarters_to_expiry <= 8:
                su_distribution[3] += su_entry.amount
        
        if total_su > 0:
            for i in range(4):
                obs[idx + i] = su_distribution[i] / total_su
        idx += 4
        
        # 成本与奖励状态 (5维)
        # 累计燃料成本 (1维)
        total_fuel_cost = sum(ship.fuel_costs)
        obs[idx] = np.log(1 + total_fuel_cost) / 25.0
        idx += 1
        
        # 累计改造CAPEX (1维)
        obs[idx] = np.log(1 + ship.retrofit_capex) / 20.0
        idx += 1
        
        # 累计改造机会成本 (1维)
        total_opportunity_cost = sum(ship.opportunity_costs)
        obs[idx] = np.log(1 + total_opportunity_cost) / 20.0
        idx += 1
        
        # 累计合规成本 (1维)
        total_compliance_cost = sum(ship.compliance_costs.values())
        obs[idx] = np.log(1 + total_compliance_cost) / 20.0
        idx += 1
        
        # 累计ZNZ奖励 (1维)
        total_znz_reward = sum(ship.znz_rewards.values())
        obs[idx] = np.log(1 + total_znz_reward) / 20.0
        idx += 1
        
        # 时间编码向量 (12维)
        # 绝对时间 (4维)
        obs[idx] = self.current_quarter / sys_config.END_QUARTER  # 整体进度
        idx += 1
        
        current_year = quarter_to_year(self.current_quarter)
        obs[idx] = (current_year - sys_config.START_YEAR) / (sys_config.END_YEAR - sys_config.START_YEAR)  # 年度进度
        idx += 1
        
        # 季度周期 (2维)
        quarter_in_year = get_quarter_in_year(self.current_quarter)
        obs[idx] = np.sin(2 * np.pi * quarter_in_year / 4)
        obs[idx + 1] = np.cos(2 * np.pi * quarter_in_year / 4)
        idx += 2
        
        # 政策时间 (4维)
        # 政策阶段 (1维)
        obs[idx] = 1.0 if current_year >= sys_config.POLICY_EFFECTIVE_START else 0.0
        idx += 1
        
        # 当前Base目标强度 (1维)
        base_intensity = policy_config.get_base_intensity(current_year)
        obs[idx] = base_intensity / 100.0
        idx += 1
        
        # 当前Direct目标强度 (1维)
        direct_intensity = policy_config.get_direct_intensity(current_year)
        obs[idx] = direct_intensity / 100.0
        idx += 1
        
        # 距项目结束时间 (1维)
        obs[idx] = (sys_config.END_QUARTER - self.current_quarter) / sys_config.END_QUARTER
        idx += 1
        
        # 约束时间 (4维)
        # 改造窗口剩余 (1维)
        latest_start = sys_config.END_QUARTER - ship.retrofit_duration + 1
        remaining_window = max(0, latest_start - self.current_quarter)
        obs[idx] = remaining_window / sys_config.END_QUARTER
        idx += 1
        
        # 技术成熟度因子 (1维)
        tech_factor = max(0.7, 1 - sys_config.TECH_IMPROVEMENT_RATE * (current_year - 2015))
        obs[idx] = tech_factor
        idx += 1
        
        # 通胀调整因子 (1维)
        inflation_factor = (1 + sys_config.INFLATION_RATE) ** (current_year - 2015)
        obs[idx] = inflation_factor / 2.0  # 归一化
        idx += 1
        
        # 季度贴现因子 (1维)
        discount_factor = get_discount_factor(self.current_quarter)
        obs[idx] = discount_factor
        
        return obs
    
    def _get_ship_observation_optimized(self, ship: Ship) -> np.ndarray:
        """获取船舶观测向量 (36维) - 优化版本，使用缓存的时间编码"""
        obs = np.zeros(36, dtype=np.float32)
        
        # 船舶状态向量 (23维) - 复用_get_ship_state_only方法
        ship_state = self._get_ship_state_only(ship)
        obs[:23] = ship_state
        
        # 成本与奖励状态 (5维)
        idx = 23
        
        # 累计燃料成本 (1维)
        total_fuel_cost = sum(ship.fuel_costs)
        obs[idx] = np.log(1 + total_fuel_cost) / 25.0
        idx += 1
        
        # 累计改造CAPEX (1维)
        obs[idx] = np.log(1 + ship.retrofit_capex) / 20.0
        idx += 1
        
        # 累计改造机会成本 (1维)
        total_opportunity_cost = sum(ship.opportunity_costs)
        obs[idx] = np.log(1 + total_opportunity_cost) / 20.0
        idx += 1
        
        # 累计合规成本 (1维)
        total_compliance_cost = sum(ship.compliance_costs.values())
        obs[idx] = np.log(1 + total_compliance_cost) / 20.0
        idx += 1
        
        # 累计ZNZ奖励 (1维)
        total_znz_reward = sum(ship.znz_rewards.values())
        obs[idx] = np.log(1 + total_znz_reward) / 20.0
        idx += 1
        
        # 时间编码向量 (8维) - 使用缓存的时间编码的前8维
        obs[28:36] = self._cached_time_encoding[:8]
        
        return obs
    
    def _compute_time_encoding(self) -> np.ndarray:
        """计算时间编码向量 (12维)"""
        time_encoding = np.zeros(12, dtype=np.float32)
        idx = 0
        
        # 绝对时间 (4维)
        time_encoding[idx] = self.current_quarter / sys_config.END_QUARTER  # 整体进度
        idx += 1
        
        current_year = quarter_to_year(self.current_quarter)
        time_encoding[idx] = (current_year - sys_config.START_YEAR) / (sys_config.END_YEAR - sys_config.START_YEAR)  # 年度进度
        idx += 1
        
        # 季度周期 (2维)
        quarter_in_year = get_quarter_in_year(self.current_quarter)
        time_encoding[idx] = np.sin(2 * np.pi * quarter_in_year / 4)
        time_encoding[idx + 1] = np.cos(2 * np.pi * quarter_in_year / 4)
        idx += 2
        
        # 政策时间 (4维)
        # 政策阶段 (1维)
        time_encoding[idx] = 1.0 if current_year >= sys_config.POLICY_EFFECTIVE_START else 0.0
        idx += 1
        
        # 当前Base目标强度 (1维)
        base_intensity = policy_config.get_base_intensity(current_year)
        time_encoding[idx] = base_intensity / 100.0
        idx += 1
        
        # 当前Direct目标强度 (1维)
        direct_intensity = policy_config.get_direct_intensity(current_year)
        time_encoding[idx] = direct_intensity / 100.0
        idx += 1
        
        # 距项目结束时间 (1维)
        time_encoding[idx] = (sys_config.END_QUARTER - self.current_quarter) / sys_config.END_QUARTER
        idx += 1
        
        # 约束时间 (4维) - 这部分对每艘船可能不同，暂时用平均值
        # 技术成熟度因子 (1维)
        tech_factor = max(0.7, 1 - sys_config.TECH_IMPROVEMENT_RATE * (current_year - 2015))
        time_encoding[idx] = tech_factor
        idx += 1
        
        # 通胀调整因子 (1维)
        inflation_factor = (1 + sys_config.INFLATION_RATE) ** (current_year - 2015)
        time_encoding[idx] = inflation_factor / 2.0  # 归一化
        idx += 1
        
        # 季度贴现因子 (1维)
        discount_factor = get_discount_factor(self.current_quarter)
        time_encoding[idx] = discount_factor
        idx += 1
        
        # 改造窗口剩余 (1维) - 使用平均改造持续时间
        avg_retrofit_duration = 8  # 平均改造持续时间
        latest_start = sys_config.END_QUARTER - avg_retrofit_duration + 1
        remaining_window = max(0, latest_start - self.current_quarter)
        time_encoding[idx] = remaining_window / sys_config.END_QUARTER
        
        return time_encoding
    
    def get_global_observation(self) -> np.ndarray:
        """获取全局观测 (用于Critic) - 高性能简化版本"""
        active_ships = self.fleet.get_active_ships()
        
        # 预计算时间编码 (12维) - 避免重复计算
        if not hasattr(self, '_cached_time_encoding') or self._cached_quarter != self.current_quarter:
            self._cached_time_encoding = self._compute_time_encoding()
            self._cached_quarter = self.current_quarter
        
        # 简化的全局状态 (20维) - 大幅减少计算量
        global_obs = np.zeros(20, dtype=np.float32)
        idx = 0
        
        if active_ships:
            # 船队基本统计 (5维)
            global_obs[idx] = len(active_ships) / self.num_ships  # 活跃船舶比例
            idx += 1
            
            # 平均船龄
            avg_age = np.mean([ship.age_quarters for ship in active_ships])
            global_obs[idx] = avg_age / sys_config.MAX_SERVICE_LIFE
            idx += 1
            
            # 船队总能耗
            total_energy = sum(ship.quarterly_energy[-1] if ship.quarterly_energy else 0 
                              for ship in active_ships)
            global_obs[idx] = np.log(1 + total_energy) / 30.0
            idx += 1
            
            # 船队总排放量
            total_emissions = sum(ship.quarterly_emissions[-1] if ship.quarterly_emissions else 0 
                                 for ship in active_ships)
            global_obs[idx] = np.log(1 + total_emissions) / 25.0
            idx += 1
            
            # 船队加权平均排放强度
            if total_energy > 0:
                weighted_intensity = total_emissions * 1000 / total_energy
                global_obs[idx] = weighted_intensity / 100.0
            idx += 1
            
            # 主要船型分布 (3维)
            type_counts = {'F': 0, 'G': 0, 'other': 0}
            for ship in active_ships:
                ship_type = ship.ship_type.value
                if ship_type in ['F']:
                    type_counts['F'] += 1
                elif ship_type in ['G', 'FG']:
                    type_counts['G'] += 1
                else:
                    type_counts['other'] += 1
            
            for ship_type in ['F', 'G', 'other']:
                global_obs[idx] = type_counts[ship_type] / len(active_ships)
                idx += 1
            
            # 改造状态统计 (3维)
            state_counts = {'operating': 0, 'retrofitting': 0, 'retired': 0}
            for ship in active_ships:
                state_counts[ship.state.value] += 1
            
            for state in ['operating', 'retrofitting', 'retired']:
                global_obs[idx] = state_counts[state] / len(active_ships)
                idx += 1
        
        # 填充剩余维度
        while idx < 20:
            global_obs[idx] = 0.0
            idx += 1
        
        # 使用缓存的时间编码 (12维)
        time_encoding = self._cached_time_encoding
        
        # 大幅简化：只使用前10艘船的简化状态 (10×5=50维)
        ship_states = []
        sample_ships = active_ships[:min(10, len(active_ships))]
        
        for ship in sample_ships:
            # 只计算最关键的5个状态特征
            state = [
                ship.age_quarters / sys_config.MAX_SERVICE_LIFE,  # 船龄
                1.0 if ship.state.value == 'operating' else 0.0,  # 运营状态
                ship.quarterly_energy[-1] / 1000.0 if ship.quarterly_energy else 0.0,  # 能耗
                ship.quarterly_emissions[-1] / 1000.0 if ship.quarterly_emissions else 0.0,  # 排放
                float(ship.teu_class[1:]) / 9.0  # TEU级别
            ]
            ship_states.extend(state)
        
        # 填充到固定长度 (50维)
        while len(ship_states) < 50:
            ship_states.append(0.0)
        
        # 组合全局观测 (20+12+50=82维)
        full_global_obs = np.concatenate([
            global_obs,      # 20维
            time_encoding,   # 12维
            ship_states[:50] # 50维
        ])
        
        return full_global_obs.astype(np.float32)
    
    def _get_ship_state_only(self, ship: Ship) -> np.ndarray:
        """获取船舶状态向量的前23维（不包含时间编码）"""
        obs = np.zeros(23, dtype=np.float32)
        
        # 船舶状态向量 (23维)
        idx = 0
        
        # 基础属性 (5维)
        # TEU级别 (1维)
        teu_level = int(ship.teu_class[1])  # C1->1, C2->2, ..., C9->9
        obs[idx] = teu_level / 9.0  # 归一化到[1/9, 1]
        idx += 1
        
        # 船舶类型 (4维) One-hot编码
        type_mapping = {'F': 0, 'G': 1, 'FG': 2, 'Z_DF': 3}
        type_idx = type_mapping.get(ship.ship_type.value, 0)
        obs[idx + type_idx] = 1.0
        idx += 4
        
        # 船队模块状态 (8维)
        # 船龄 (1维)
        obs[idx] = ship.age_quarters / sys_config.MAX_SERVICE_LIFE
        idx += 1
        
        # 改造状态 (3维) One-hot编码
        state_mapping = {'operating': 0, 'retrofitting': 1, 'retired': 2}
        state_idx = state_mapping.get(ship.state.value, 0)
        if state_idx < 3:
            obs[idx + state_idx] = 1.0
        idx += 3
        
        # 季度能耗 (1维)
        if ship.quarterly_energy:
            energy = ship.quarterly_energy[-1]
            obs[idx] = np.log(1 + energy) / 20.0  # 对数归一化
        idx += 1
        
        # 季度排放量 (1维)
        if ship.quarterly_emissions:
            emissions = ship.quarterly_emissions[-1]
            obs[idx] = np.log(1 + emissions) / 15.0  # 对数归一化
        idx += 1
        
        # 年度能耗累计 (1维)
        current_year = quarter_to_year(self.current_quarter)
        annual_energy = ship.annual_energy.get(current_year, 0)
        quarter_in_year = get_quarter_in_year(self.current_quarter)
        if quarter_in_year > 0 and annual_energy > 0:
            expected_annual = annual_energy / quarter_in_year * 4
            obs[idx] = annual_energy / expected_annual
        else:
            obs[idx] = 0.0
        idx += 1
        
        # 年度排放累计 (1维)
        annual_emissions = ship.annual_emissions.get(current_year, 0)
        if quarter_in_year > 0 and annual_emissions > 0:
            expected_annual_emissions = annual_emissions / quarter_in_year * 4
            obs[idx] = annual_emissions / expected_annual_emissions
        else:
            obs[idx] = 0.0
        idx += 1
        
        # 年度排放强度 (1维)
        annual_intensity = ship.annual_emission_intensity.get(current_year, 0)
        obs[idx] = annual_intensity / 100.0  # 归一化到[0,1]
        idx += 1
        
        # SU管理状态 (5维)
        # 当前SU库存总量 (1维)
        total_su = ship.get_available_su()
        obs[idx] = np.log(1 + total_su) / 10.0
        idx += 1
        
        # SU库存按到期时间分布 (4维)
        su_distribution = [0, 0, 0, 0]  # [1-2季度, 3-4季度, 5-6季度, 7-8季度]
        for su_entry in ship.su_inventory:
            quarters_to_expiry = su_entry.expiry_quarter - self.current_quarter
            if 1 <= quarters_to_expiry <= 2:
                su_distribution[0] += su_entry.amount
            elif 3 <= quarters_to_expiry <= 4:
                su_distribution[1] += su_entry.amount
            elif 5 <= quarters_to_expiry <= 6:
                su_distribution[2] += su_entry.amount
            elif 7 <= quarters_to_expiry <= 8:
                su_distribution[3] += su_entry.amount
        
        # 归一化SU分布
        total_distributed = sum(su_distribution)
        if total_distributed > 0:
            for i in range(4):
                obs[idx + i] = su_distribution[i] / total_distributed
        idx += 4
        
        return obs
    
    def _record_metrics(self, rewards: Dict[int, float]):
        """记录性能指标"""
        total_reward = sum(rewards.values())
        total_emissions = sum(
            ship.quarterly_emissions[-1] if ship.quarterly_emissions else 0 
            for ship in self.fleet.get_active_ships()
        )
        
        self.episode_rewards.append(total_reward)
        self.episode_emissions.append(total_emissions)
        
        # 更新总现值成本
        self.total_pv_cost -= total_reward  # 奖励是负成本
        
        # 记录季度指标
        metrics = {
            'quarter': self.current_quarter,
            'year': quarter_to_year(self.current_quarter),
            'total_reward': total_reward,
            'total_emissions': total_emissions,
            'active_ships': len(self.fleet.get_active_ships()),
            'total_pv_cost': self.total_pv_cost
        }
        
        self.quarterly_metrics.append(metrics)
    
    def get_episode_summary(self) -> Dict:
        """获取回合总结"""
        return {
            'total_quarters': len(self.episode_rewards),
            'total_reward': sum(self.episode_rewards),
            'total_emissions': sum(self.episode_emissions),
            'total_pv_cost': self.total_pv_cost,
            'average_quarterly_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'average_quarterly_emissions': np.mean(self.episode_emissions) if self.episode_emissions else 0,
            'final_fleet_stats': self.fleet.get_fleet_statistics(),
            'quarterly_metrics': self.quarterly_metrics
        }

if __name__ == "__main__":
    # 测试功能已移除 - 请使用真实船舶数据
    print("请使用 load_real_ships.py 加载真实船舶数据后创建环境")
    print("示例:")
    print("from load_real_ships import load_ships_from_excel")
    print("fleet = load_ships_from_excel('Container_ship.xlsx', 10)")
    print("env = MaritimeDecarbonizationEnv(num_ships=10, random_seed=42, fleet=fleet)")
