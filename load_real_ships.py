import pandas as pd
import numpy as np
from typing import List
from ship import Ship, Fleet, ShipType
from datetime import datetime
import random

def load_ships_from_excel(excel_path: str, num_ships: int = None) -> Fleet:
    """
    从Excel文件加载真实船舶数据创建船队
    
    Args:
        excel_path: Excel文件路径
        num_ships: 要加载的船舶数量，如果为None则使用所有船舶
    
    Returns:
        Fleet: 包含真实船舶数据的船队
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        print(f"成功读取 {len(df)} 条船舶记录")
        
        # 如果num_ships为None或者大于等于数据总量，使用所有船舶
        if num_ships is None or num_ships >= len(df):
            df_sample = df.reset_index(drop=True)
            print(f"使用所有 {len(df_sample)} 艘船舶")
        else:
            # 如果数据量不足，重复使用
            if len(df) < num_ships:
                print(f"数据量不足，将重复使用现有数据")
                # 重复数据直到达到所需数量
                repeat_times = (num_ships // len(df)) + 1
                df = pd.concat([df] * repeat_times, ignore_index=True)
            
            # 完全随机抽取指定数量的船舶
            df_sample = df.sample(n=num_ships, random_state=42).reset_index(drop=True)
            print(f"随机抽取了 {len(df_sample)} 艘船舶")
        
        # 显示燃料类型分布
        fuel_distribution = df_sample['Fuel_Type'].value_counts()
        print("抽取船舶的燃料类型分布:")
        for fuel_type, count in fuel_distribution.items():
            print(f"  {fuel_type}: {count}艘")
        
        fleet = Fleet()
        current_year = datetime.now().year
        
        for idx, row in df_sample.iterrows():
            ship_id = idx + 1
            
            # 处理船舶容量 (Size字段，单位TEU)
            teu_capacity = int(row['Size']) if pd.notna(row['Size']) else 1000
            
            # 根据TEU容量确定TEU级别
            teu_class = determine_teu_class(teu_capacity)
            
            # 处理燃料类型
            fuel_type = row['Fuel_Type'] if pd.notna(row['Fuel_Type']) else 'HFO'
            ship_type = determine_ship_type(fuel_type)
            
            # 计算船龄（季度）
            build_year = int(row['Build_Year']) if pd.notna(row['Build_Year']) else 2010
            ship_age_years = current_year - build_year
            age_quarters = max(0, min(120, ship_age_years * 4))  # 限制在0-120季度
            
            # 处理真实主机功率 (ME字段，单位kW)
            real_engine_power = None
            if pd.notna(row['ME']) and row['ME'] > 0:
                real_engine_power = float(row['ME'])
            
            # 创建船舶对象，传入真实功率
            ship = Ship(
                ship_id=ship_id,
                teu_class=teu_class,
                ship_type=ship_type,
                teu_capacity=teu_capacity,
                age_quarters=age_quarters,
                real_engine_power=real_engine_power
            )
            
            # 添加真实船舶的额外信息
            ship.real_data = {
                'name': row['Name'] if pd.notna(row['Name']) else f'Ship_{ship_id}',
                'imo': row['IMO'] if pd.notna(row['IMO']) else None,
                'mmsi': row['MMSI'] if pd.notna(row['MMSI']) else None,
                'build_year': build_year,
                'dwt': row['DWT'] if pd.notna(row['DWT']) else None,
                'engine_power_original': row['ME'] if pd.notna(row['ME']) else None,
                'fuel_type_original': fuel_type,
                'using_real_power': ship.using_real_power,
                'actual_engine_power': ship.engine_power
            }
            
            fleet.add_ship(ship)
        
        print(f"成功创建包含 {len(fleet.ships)} 艘真实船舶的船队")
        
        # 打印船队统计
        stats = fleet.get_fleet_statistics()
        print("\n船队统计信息:")
        print(f"总船舶数: {stats['total_ships']}")
        print("\nTEU级别分布:")
        for teu_class, count in stats['teu_distribution'].items():
            print(f"  {teu_class}: {count}艘")
        print("\n船舶类型分布:")
        for ship_type, count in stats['type_distribution'].items():
            print(f"  {ship_type}: {count}艘")
        
        return fleet
        
    except Exception as e:
        print(f"加载船舶数据时出错: {e}")
        # 不再提供随机生成的备选方案
        raise Exception("无法加载真实船舶数据，请检查Excel文件路径和格式")

def load_ships_from_csv(csv_path: str, num_ships: int = 100) -> Fleet:
    """
    从CSV文件加载真实船舶数据创建船队
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条船舶记录")
        
        # 如果数据量不足，重复使用
        if len(df) < num_ships:
            print(f"数据量不足，将重复使用现有数据")
            # 重复数据直到达到所需数量
            repeat_times = (num_ships // len(df)) + 1
            df = pd.concat([df] * repeat_times, ignore_index=True)
        
        # 如果需要的船舶数量少于CSV中的数量，则取前num_ships艘
        if len(df) > num_ships:
            df = df.head(num_ships).reset_index(drop=True)
        
        print(f"使用了 {len(df)} 艘船舶")
        
        fleet = Fleet()
        current_year = datetime.now().year
        
        for idx, row in df.iterrows():
            ship_id = idx + 1
            
            # 从CSV读取数据
            teu_capacity = int(row['teu_capacity'])
            teu_class = row['teu_class']
            ship_type_str = row['ship_type']
            age_quarters = int(row['age_quarters'])
            build_year = int(row['build_year'])
            fuel_type_original = row['fuel_type_original']
            name = row['name']
            
            # 确定船舶类型
            if ship_type_str == 'F':
                ship_type = ShipType.FOSSIL_FUEL
            elif ship_type_str == 'G':
                ship_type = ShipType.GREEN_FUEL
            elif ship_type_str == 'Z':
                ship_type = ShipType.ZERO_EMISSION
            else:
                ship_type = ShipType.FOSSIL_FUEL  # 默认
            
            # 创建船舶对象
            ship = Ship(
                ship_id=ship_id,
                teu_capacity=teu_capacity,
                teu_class=teu_class,
                ship_type=ship_type,
                age_quarters=age_quarters
            )
            
            # 添加真实数据
            ship.real_data = {
                'name': name,
                'build_year': build_year,
                'fuel_type_original': fuel_type_original
            }
            
            fleet.add_ship(ship)
        
        print(f"成功创建包含 {len(fleet.ships)} 艘真实船舶的船队")
        
        # 打印船队统计
        stats = fleet.get_fleet_statistics()
        print("\n船队统计信息:")
        print(f"总船舶数: {stats['total_ships']}")
        print("\nTEU级别分布:")
        for teu_class, count in stats['teu_distribution'].items():
            print(f"  {teu_class}: {count}艘")
        print("\n船舶类型分布:")
        for ship_type, count in stats['type_distribution'].items():
            print(f"  {ship_type}: {count}艘")
        
        return fleet
        
    except Exception as e:
        print(f"加载船舶数据时出错: {e}")
        # 不再提供随机生成的备选方案
        raise Exception("无法加载真实船舶数据，请检查CSV文件路径和格式")

def determine_teu_class(teu_capacity: int) -> str:
    """
    根据TEU容量确定TEU级别
    """
    if teu_capacity < 500:
        return 'C1'
    elif teu_capacity < 1000:
        return 'C2'
    elif teu_capacity < 1500:
        return 'C3'
    elif teu_capacity < 2500:
        return 'C4'
    elif teu_capacity < 4000:
        return 'C5'
    elif teu_capacity < 6000:
        return 'C6'
    elif teu_capacity < 10000:
        return 'C7'
    elif teu_capacity < 15000:
        return 'C8'
    else:
        return 'C9'

def determine_ship_type(fuel_type: str) -> ShipType:
    """
    根据燃料类型确定船舶类型
    
    参数:
        fuel_type: 燃料类型字符串
        
    返回:
        ShipType: 对应的船舶类型枚举
        
    燃料类型映射:
        - HFO, MDO, FOSSIL_FUEL, DIESEL, FUEL_OIL -> F类 (传统燃料船)
        - LNG, NATURAL_GAS -> G类 (LNG专用船)
        - DUAL_FUEL, HYBRID -> FG类 (双燃料船)
        - METHANOL, AMMONIA, HYDROGEN -> Z_DF类 (甲醇双燃料船)
    """
    if not fuel_type or pd.isna(fuel_type):
        print(f"警告: 燃料类型为空，默认设为F类")
        return ShipType.F
    
    fuel_type = str(fuel_type).upper().strip()
    
    # 传统燃料 -> F类
    if fuel_type in ['HFO', 'MDO', 'FOSSIL_FUEL', 'DIESEL', 'FUEL_OIL', 'HEAVY_FUEL_OIL', 'MARINE_DIESEL']:
        return ShipType.F
    
    # LNG燃料 -> G类
    elif fuel_type in ['LNG', 'NATURAL_GAS', 'LIQUEFIED_NATURAL_GAS']:
        return ShipType.G
    
    # 双燃料 -> FG类
    elif fuel_type in ['DUAL_FUEL', 'HYBRID', 'LNG_DUAL_FUEL']:
        return ShipType.FG
    
    # 甲醇等新燃料 -> Z_DF类
    elif fuel_type in ['METHANOL', 'AMMONIA', 'HYDROGEN', 'GREEN_METHANOL']:
        return ShipType.Z_DF
    
    else:
        print(f"警告: 未知燃料类型 '{fuel_type}'，默认设为F类")
        return ShipType.F

if __name__ == "__main__":
    # 测试加载函数
    csv_path = "sampled_ships.csv"
    fleet = load_ships_from_csv(csv_path, 100)
    
    # 显示前几艘船的详细信息
    print("\n前5艘船的详细信息:")
    for i, ship in enumerate(fleet.ships[:5]):
        print(f"\n船舶 {ship.ship_id}:")
        print(f"  名称: {ship.real_data['name']}")
        print(f"  TEU容量: {ship.teu_capacity}")
        print(f"  TEU级别: {ship.teu_class}")
        print(f"  船舶类型: {ship.ship_type.value}")
        print(f"  船龄: {ship.age_quarters//4}年 ({ship.age_quarters}季度)")
        print(f"  建造年份: {ship.real_data['build_year']}")
        print(f"  原始燃料类型: {ship.real_data['fuel_type_original']}")