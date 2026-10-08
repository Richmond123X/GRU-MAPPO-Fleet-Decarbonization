"""船舶模块 - 船舶数据结构和船队管理"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from config import (
    sys_config, teu_config, fuel_config, policy_config, 
    get_discount_factor, quarter_to_year, get_quarter_in_year,
    calculate_mixed_fuel_properties
)

class ShipType(Enum):
    """船舶类型枚举"""
    F = "F"      # 传统燃料船
    G = "G"      # LNG专用船
    FG = "FG"    # LNG双燃料船
    Z_DF = "Z_DF"  # 甲醇双燃料船

class ShipState(Enum):
    """船舶状态枚举"""
    OPERATING = "operating"      # 运营中
    RETROFITTING = "retrofitting"  # 改造中
    RETIRED = "retired"          # 已退役

@dataclass
class SUEntry:
    """SU库存条目"""
    amount: float  # SU数量 (tCO₂e)
    expiry_quarter: int  # 到期季度
    
class Ship:
    """船舶类"""
    
    def __init__(self, ship_id: int, teu_class: str, ship_type: ShipType, 
                 teu_capacity: int, age_quarters: int = 0, real_engine_power: float = None):
        self.ship_id = ship_id
        self.teu_class = teu_class
        self.ship_type = ship_type
        self.teu_capacity = teu_capacity
        self.age_quarters = age_quarters
        
        # 从配置获取船舶属性
        teu_info = teu_config.TEU_CLASSES[teu_class]
        self.retrofit_duration = teu_info['retrofit_duration']
        self.daily_charter_rate = teu_info['daily_charter']
        
        # 优先使用真实功率，如果没有则计算功率
        if real_engine_power is not None and real_engine_power > 0:
            self.engine_power = real_engine_power
            self.using_real_power = True
        else:
            # 根据TEU容量计算功率 (kW) - 功率 = 4 * 容量(TEU)
            self.engine_power = 4.0 * teu_capacity
            self.using_real_power = False
        
        # 状态管理
        self.state = ShipState.OPERATING
        self.retrofit_progress = 0  # 改造进度 (季度)
        self.retrofit_start_quarter = None
        
        # 能耗和排放历史
        self.quarterly_energy = []  # 季度能耗历史 (GJ)
        self.quarterly_emissions = []  # 季度排放历史 (tCO₂e)
        self.annual_energy = {}  # 年度能耗累计 {year: GJ}
        self.annual_emissions = {}  # 年度排放累计 {year: tCO₂e}
        self.annual_emission_intensity = {}  # 年度排放强度 {year: gCO₂e/MJ}
        
        # SU管理
        self.su_inventory: List[SUEntry] = []  # SU库存
        self.su_history = {}  # SU产生历史 {year: amount}
        
        # 成本和收益历史
        self.fuel_costs = []  # 季度燃料成本
        self.retrofit_capex = 0  # 改造资本支出
        self.opportunity_costs = []  # 机会成本历史
        self.compliance_costs = {}  # 年度合规成本 {year: cost}
        self.znz_rewards = {}  # 年度ZNZ奖励 {year: reward}
        self.su_sale_revenues = {}  # 年度SU出售收益 {year: revenue}
        
    def get_quarterly_energy_baseline(self, quarter_in_year: int) -> float:
        """获取季度基准能耗"""
        baseline = teu_config.QUARTERLY_ENERGY_BASELINE[self.teu_class]
        return baseline[quarter_in_year - 1]  # quarter_in_year是1-4
    
    def calculate_quarterly_energy(self, current_quarter: int) -> float:
        """计算季度能耗"""
        if self.state != ShipState.OPERATING:
            return 0.0
            
        year = quarter_to_year(current_quarter)
        quarter_in_year = get_quarter_in_year(current_quarter)
        
        # 基准能耗
        baseline_energy = self.get_quarterly_energy_baseline(quarter_in_year)
        
        # 年度增长调整
        growth_factor = (1 + sys_config.ENERGY_GROWTH_RATE) ** (year - 2024)
        
        # 运营系数 (运营中为1.0，其他为0.0)
        operation_factor = 1.0 if self.state == ShipState.OPERATING else 0.0
        
        energy = baseline_energy * growth_factor * operation_factor
        return energy
    
    def calculate_quarterly_emissions(self, energy: float, fuel_strategy: str) -> float:
        """计算季度排放量"""
        if energy == 0:
            return 0.0
            
        # 获取燃料策略的排放因子
        emission_factor = self.get_fuel_emission_factor(fuel_strategy)
        
        # 排放量 = 能耗 × 排放因子 / 1000 (GJ × gCO₂e/MJ × 1000MJ/GJ / 1000 = tCO₂e)
        emissions = energy * emission_factor / 1000
        return emissions
    
    def get_fuel_emission_factor(self, fuel_strategy: str) -> float:
        """获取燃料策略的排放因子"""
        if self.ship_type == ShipType.F:
            if fuel_strategy in fuel_config.F_CLASS_STRATEGIES:
                fuel_mix = fuel_config.F_CLASS_STRATEGIES[fuel_strategy]
                emission_factor, _ = calculate_mixed_fuel_properties(fuel_mix)
                return emission_factor
        elif self.ship_type in [ShipType.G, ShipType.FG]:
            if fuel_strategy in fuel_config.G_CLASS_STRATEGIES:
                fuel_mix = fuel_config.G_CLASS_STRATEGIES[fuel_strategy]
                emission_factor, _ = calculate_mixed_fuel_properties(fuel_mix)
                return emission_factor
        elif self.ship_type == ShipType.Z_DF:
            if fuel_strategy in fuel_config.Z_CLASS_STRATEGIES:
                fuel_mix = fuel_config.Z_CLASS_STRATEGIES[fuel_strategy]
                emission_factor, _ = calculate_mixed_fuel_properties(fuel_mix)
                return emission_factor
        
        # 默认返回传统燃料排放因子
        return fuel_config.BASE_FUELS['F']['emission_factor']
    
    def get_fuel_price(self, fuel_strategy: str) -> float:
        """获取燃料策略的价格"""
        if self.ship_type == ShipType.F:
            if fuel_strategy in fuel_config.F_CLASS_STRATEGIES:
                fuel_mix = fuel_config.F_CLASS_STRATEGIES[fuel_strategy]
                _, price = calculate_mixed_fuel_properties(fuel_mix)
                return price
        elif self.ship_type in [ShipType.G, ShipType.FG]:
            if fuel_strategy in fuel_config.G_CLASS_STRATEGIES:
                fuel_mix = fuel_config.G_CLASS_STRATEGIES[fuel_strategy]
                _, price = calculate_mixed_fuel_properties(fuel_mix)
                return price
        elif self.ship_type == ShipType.Z_DF:
            if fuel_strategy in fuel_config.Z_CLASS_STRATEGIES:
                fuel_mix = fuel_config.Z_CLASS_STRATEGIES[fuel_strategy]
                _, price = calculate_mixed_fuel_properties(fuel_mix)
                return price
        
        # 默认返回传统燃料价格
        return fuel_config.BASE_FUELS['F']['price']
    
    def update_quarterly_data(self, current_quarter: int, fuel_strategy: str):
        """更新季度数据"""
        # 计算能耗和排放
        energy = self.calculate_quarterly_energy(current_quarter)
        emissions = self.calculate_quarterly_emissions(energy, fuel_strategy)
        
        # 记录季度数据
        self.quarterly_energy.append(energy)
        self.quarterly_emissions.append(emissions)
        
        # 计算燃料成本 (GJ × USD/GJ = USD)
        fuel_price = self.get_fuel_price(fuel_strategy)
        fuel_cost = energy * fuel_price
        self.fuel_costs.append(fuel_cost)
        
        # 更新年度累计
        year = quarter_to_year(current_quarter)
        if year not in self.annual_energy:
            self.annual_energy[year] = 0
            self.annual_emissions[year] = 0
        
        self.annual_energy[year] += energy
        self.annual_emissions[year] += emissions
        
        # 计算年度排放强度 (tCO₂e × 1000 / GJ = gCO₂e/MJ)
        if self.annual_energy[year] > 0:
            self.annual_emission_intensity[year] = (
                self.annual_emissions[year] * 1000 / self.annual_energy[year]
            )
        else:
            self.annual_emission_intensity[year] = 0
    
    def update_su_inventory(self, current_quarter: int):
        """更新SU库存 - 移除过期SU"""
        self.su_inventory = [
            su for su in self.su_inventory 
            if su.expiry_quarter > current_quarter
        ]
    
    def calculate_annual_su_ru(self, year: int) -> Tuple[float, float, float]:
        """计算年度SU和RU"""
        if year < sys_config.POLICY_EFFECTIVE_START:
            return 0.0, 0.0, 0.0
        
        if year not in self.annual_emission_intensity:
            return 0.0, 0.0, 0.0
        
        annual_energy = self.annual_energy.get(year, 0)
        annual_intensity = self.annual_emission_intensity.get(year, 0)
        
        if annual_energy == 0:
            return 0.0, 0.0, 0.0
        
        base_intensity = policy_config.get_base_intensity(year)
        direct_intensity = policy_config.get_direct_intensity(year)
        
        # 计算SU (盈余单位)
        su = 0.0
        if annual_intensity < direct_intensity:
            su = (direct_intensity - annual_intensity) * annual_energy / 1000
        
        # 计算RU1 (一级赤字单位)
        ru1 = 0.0
        if annual_intensity > direct_intensity:
            # RU1覆盖从direct_intensity到base_intensity的范围
            ru1 = (min(base_intensity, annual_intensity) - direct_intensity) * annual_energy / 1000
        
        # 计算RU2 (二级赤字单位)
        ru2 = 0.0
        if annual_intensity > base_intensity:
            # RU2覆盖超过base_intensity的部分
            ru2 = (annual_intensity - base_intensity) * annual_energy / 1000
        
        return su, ru1, ru2
    
    def add_su_to_inventory(self, amount: float, current_quarter: int):
        """添加SU到库存"""
        if amount > 0:
            expiry_quarter = current_quarter + sys_config.SU_VALIDITY_PERIOD
            self.su_inventory.append(SUEntry(amount, expiry_quarter))
    
    def get_available_su(self) -> float:
        """获取可用SU总量"""
        return sum(su.amount for su in self.su_inventory)
    
    def use_su_for_ru(self, ru_demand: float) -> Tuple[float, float]:
        """使用SU抵消RU需求 (FIFO原则)"""
        used_su = 0.0
        remaining_ru = ru_demand
        
        # 按到期时间排序 (FIFO)
        self.su_inventory.sort(key=lambda x: x.expiry_quarter)
        
        i = 0
        while i < len(self.su_inventory) and remaining_ru > 0:
            su_entry = self.su_inventory[i]
            if su_entry.amount <= remaining_ru:
                # 完全使用这个SU条目
                used_su += su_entry.amount
                remaining_ru -= su_entry.amount
                self.su_inventory.pop(i)
            else:
                # 部分使用这个SU条目
                used_su += remaining_ru
                su_entry.amount -= remaining_ru
                remaining_ru = 0
                i += 1
        
        return used_su, remaining_ru
    
    def start_retrofit(self, current_quarter: int) -> bool:
        """启动改造 - 硬性约束：只有F类运营中的船舶可以开始改装"""
        # 硬性约束1：只有F类船舶可以改装
        if self.ship_type != ShipType.F:
            return False
        
        # 硬性约束2：只有运营中的船舶可以改装
        if self.state != ShipState.OPERATING:
            return False
        
        # 硬性约束3：检查改装时机约束
        latest_start = sys_config.END_QUARTER - self.retrofit_duration + 1
        if current_quarter > latest_start:
            return False
        
        # 硬性约束4：不能重复改装
        if self.retrofit_start_quarter is not None:
            return False
        
        # 执行改装启动
        self.state = ShipState.RETROFITTING
        self.retrofit_start_quarter = current_quarter
        self.retrofit_progress = 0
        
        # 计算改造成本
        year = quarter_to_year(current_quarter)
        from config import retrofit_config
        self.retrofit_capex = retrofit_config.calculate_retrofit_cost(
            self.engine_power, year
        )
        
        return True
    
    def continue_retrofit(self) -> bool:
        """继续改造 - 硬性约束：只有正在改装的船舶可以继续改装"""
        # 硬性约束1：只有正在改装的船舶可以继续改装
        if self.state != ShipState.RETROFITTING:
            return False
        
        # 硬性约束2：检查改装进度不能超过规定持续时间
        if self.retrofit_progress >= self.retrofit_duration:
            # 改装已完成，不能继续
            return False
        
        # 硬性约束3：必须有改装开始记录
        if self.retrofit_start_quarter is None:
            # 没有改装开始记录，强制结束改装状态
            self.state = ShipState.OPERATING
            return False
        
        # 执行改装进度
        self.retrofit_progress += 1
        
        # 检查是否完成改造
        if self.retrofit_progress >= self.retrofit_duration:
            # 硬性约束4：改装完成后强制转换为Z_DF类
            self.state = ShipState.OPERATING
            self.ship_type = ShipType.Z_DF
            return True  # 返回True表示改装完成
        
        return False  # 返回False表示改装继续
    
    def validate_retrofit_state(self) -> bool:
        """验证改装状态的一致性 - 硬性约束验证"""
        # 验证1：改装状态与改装记录的一致性
        if self.state == ShipState.RETROFITTING:
            if self.retrofit_start_quarter is None:
                # 状态不一致，强制修复
                self.state = ShipState.OPERATING
                return False
            
            # 验证改装时间不超过规定持续时间
            if self.retrofit_progress >= self.retrofit_duration:
                # 改装时间已到，强制完成改装
                self.state = ShipState.OPERATING
                self.ship_type = ShipType.Z_DF
                return False
        
        # 验证2：Z_DF类船舶不应该有改装状态
        if self.ship_type == ShipType.Z_DF and self.state == ShipState.RETROFITTING:
            # 状态不一致，强制修复
            self.state = ShipState.OPERATING
            return False
        
        # 验证3：非F类船舶不应该有改装记录
        if self.ship_type != ShipType.F and self.retrofit_start_quarter is not None:
            # 清理不一致的改装记录
            if self.ship_type == ShipType.Z_DF:
                # Z_DF类船舶保持改装记录但确保状态正确
                self.state = ShipState.OPERATING
            else:
                # 其他类型船舶清理改装记录
                self.retrofit_start_quarter = None
                self.retrofit_progress = 0
                self.state = ShipState.OPERATING
            return False
        
        return True
    
    def calculate_opportunity_cost(self) -> float:
        """计算季度机会成本"""
        if self.state == ShipState.RETROFITTING:
            return self.daily_charter_rate * sys_config.DAYS_PER_QUARTER
        return 0.0
    
    def is_retired(self) -> bool:
        """检查是否退役"""
        return self.age_quarters >= sys_config.MAX_SERVICE_LIFE
    
    def age_one_quarter(self):
        """船龄增加一个季度"""
        self.age_quarters += 1
        if self.is_retired():
            self.state = ShipState.RETIRED
    
    def get_valid_actions(self) -> List[str]:
        """获取有效动作列表"""
        if self.state == ShipState.RETIRED:
            return []
        
        if self.state == ShipState.RETROFITTING:
            return ['RETROFIT_CONTINUE']
        
        # 运营状态的动作
        actions = []
        
        if self.ship_type == ShipType.F:
            actions.extend(['B0', 'B5', 'B7', 'B10', 'B20', 'B30', 'B50', 'B100'])
            # 检查是否可以启动改造
            # 这里简化检查，实际检查在start_retrofit中
            actions.append('START_RETROFIT')
        elif self.ship_type in [ShipType.G, ShipType.FG]:
            actions.extend(['G0', 'G5', 'G10', 'G20', 'G50'])
        elif self.ship_type == ShipType.Z_DF:
            actions.extend(['Z100'])
        
        return actions

class Fleet:
    """船队类"""
    
    def __init__(self):
        self.ships: List[Ship] = []
        self.current_quarter = 1
        
    def add_ship(self, ship: Ship):
        """添加船舶到船队"""
        self.ships.append(ship)
    
    def get_active_ships(self) -> List[Ship]:
        """获取活跃船舶 (未退役)"""
        return [ship for ship in self.ships if ship.state != ShipState.RETIRED]
    
    def get_fleet_statistics(self) -> Dict:
        """获取船队统计信息"""
        active_ships = self.get_active_ships()
        
        stats = {
            'total_ships': len(self.ships),
            'active_ships': len(active_ships),
            'retired_ships': len(self.ships) - len(active_ships),
            'teu_distribution': {},
            'type_distribution': {},
            'state_distribution': {},
            'total_energy': 0,
            'total_emissions': 0
        }
        
        # 统计分布
        for ship in active_ships:
            # TEU分布
            stats['teu_distribution'][ship.teu_class] = (
                stats['teu_distribution'].get(ship.teu_class, 0) + 1
            )
            
            # 类型分布
            stats['type_distribution'][ship.ship_type.value] = (
                stats['type_distribution'].get(ship.ship_type.value, 0) + 1
            )
            
            # 状态分布
            stats['state_distribution'][ship.state.value] = (
                stats['state_distribution'].get(ship.state.value, 0) + 1
            )
            
            # 当前季度能耗和排放
            if ship.quarterly_energy:
                stats['total_energy'] += ship.quarterly_energy[-1]
            if ship.quarterly_emissions:
                stats['total_emissions'] += ship.quarterly_emissions[-1]
        
        return stats
    
    def step(self, actions: Dict[int, str]):
        """船队前进一步 - 包含改装逻辑硬性约束"""
        self.current_quarter += 1
        
        for ship in self.get_active_ships():
            # 船龄增加
            ship.age_one_quarter()
            
            # 更新SU库存
            ship.update_su_inventory(self.current_quarter)
            
            # 硬性约束：验证改装状态一致性
            ship.validate_retrofit_state()
            
            # 执行动作
            action = actions.get(ship.ship_id, 'B0')  # 默认动作
            
            # 硬性约束：强制执行改装逻辑
            if ship.state == ShipState.RETROFITTING:
                # 正在改装的船舶只能执行RETROFIT_CONTINUE
                if action != 'RETROFIT_CONTINUE':
                    action = 'RETROFIT_CONTINUE'  # 强制改为继续改装
                
                # 执行改装继续
                retrofit_completed = ship.continue_retrofit()
                if retrofit_completed:
                    # 改装完成，记录机会成本
                    opportunity_cost = ship.calculate_opportunity_cost()
                    ship.opportunity_costs.append(opportunity_cost)
                else:
                    # 改装继续，记录机会成本
                    opportunity_cost = ship.calculate_opportunity_cost()
                    ship.opportunity_costs.append(opportunity_cost)
            
            elif action == 'START_RETROFIT':
                # 硬性约束：检查是否可以开始改装
                if ship.ship_type == ShipType.F and ship.state == ShipState.OPERATING:
                    success = ship.start_retrofit(self.current_quarter)
                    if success:
                        # 改装启动成功，记录机会成本
                        opportunity_cost = ship.calculate_opportunity_cost()
                        ship.opportunity_costs.append(opportunity_cost)
                    else:
                        # 改装启动失败，使用默认燃料策略
                        default_action = 'B0'
                        ship.update_quarterly_data(self.current_quarter, default_action)
                        opportunity_cost = ship.calculate_opportunity_cost()
                        ship.opportunity_costs.append(opportunity_cost)
                else:
                    # 不符合改装条件，使用默认燃料策略
                    default_action = 'B0' if ship.ship_type == ShipType.F else 'Z100' if ship.ship_type == ShipType.Z_DF else 'G0'
                    ship.update_quarterly_data(self.current_quarter, default_action)
                    opportunity_cost = ship.calculate_opportunity_cost()
                    ship.opportunity_costs.append(opportunity_cost)
            
            elif action == 'RETROFIT_CONTINUE':
                # 硬性约束：只有正在改装的船舶可以继续改装
                if ship.state == ShipState.RETROFITTING:
                    retrofit_completed = ship.continue_retrofit()
                    if retrofit_completed:
                        # 改装完成，记录机会成本
                        opportunity_cost = ship.calculate_opportunity_cost()
                        ship.opportunity_costs.append(opportunity_cost)
                    else:
                        # 改装继续，记录机会成本
                        opportunity_cost = ship.calculate_opportunity_cost()
                        ship.opportunity_costs.append(opportunity_cost)
                else:
                    # 不在改装状态，使用适当的燃料策略
                    if ship.ship_type == ShipType.F:
                        default_action = 'B0'
                    elif ship.ship_type == ShipType.Z_DF:
                        default_action = 'Z100'
                    else:
                        default_action = 'G0'
                    
                    ship.update_quarterly_data(self.current_quarter, default_action)
                    opportunity_cost = ship.calculate_opportunity_cost()
                    ship.opportunity_costs.append(opportunity_cost)
            
            else:
                # 燃料策略动作 - 硬性约束：验证动作有效性
                valid_actions = ship.get_valid_actions()
                
                if action not in valid_actions:
                    # 动作无效，使用默认动作
                    if ship.ship_type == ShipType.F:
                        action = 'B0'
                    elif ship.ship_type == ShipType.Z_DF:
                        action = 'Z100'
                    elif ship.ship_type in [ShipType.G, ShipType.FG]:
                        action = 'G0'
                    else:
                        action = 'B0'
                
                # 硬性约束：Z_DF类船舶只能使用Z100
                if ship.ship_type == ShipType.Z_DF and action != 'Z100':
                    action = 'Z100'
                
                # 执行燃料策略
                ship.update_quarterly_data(self.current_quarter, action)
                
                # 计算机会成本
                opportunity_cost = ship.calculate_opportunity_cost()
                ship.opportunity_costs.append(opportunity_cost)

# 随机生成船队的功能已被移除，现在只使用真实船舶数据

if __name__ == "__main__":
    # 测试功能已移除 - 请使用 load_real_ships.py 加载真实数据
    print("请使用 load_real_ships.py 中的函数加载真实船舶数据")