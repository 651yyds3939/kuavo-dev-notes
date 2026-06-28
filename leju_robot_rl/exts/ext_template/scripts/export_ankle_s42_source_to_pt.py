#!/usr/bin/env python3
"""
将 ankle_s42_source.py 中的 TorchScript 函数导出为 .pt 文件

使用方法:
python export_ankle_s42_source_to_pt.py --output_dir ./actuators
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加模块搜索路径
current_dir = Path(__file__).parent
actuators_dir = current_dir.parent / "ext_template" / "actuators"
sys.path.insert(0, str(actuators_dir))

# 导入 ankle_s42_source 模块
import ankle_s42_source


def export_ankle_s42_functions(output_dir: str):
    """
    导出 ankle_s42_source.py 中的所有 TorchScript 函数为 .pt 文件
    
    Args:
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在导出 ankle_s42_source 函数到: {output_dir}")
    
    # 需要导出的函数列表 - 这些函数已经用 @torch.jit.script 装饰
    functions_to_export = [
        ('joint_to_motor_position', ankle_s42_source.joint_to_motor_position),
        ('get_left_ankle_matrix', ankle_s42_source.get_left_ankle_matrix),
        ('get_knee_matrix', ankle_s42_source.get_knee_matrix),
        ('get_right_ankle_matrix', ankle_s42_source.get_right_ankle_matrix),
        ('joint_to_motor_velocity', ankle_s42_source.joint_to_motor_velocity),
        ('motor_to_joint_torque', ankle_s42_source.motor_to_joint_torque),
        ('get_joint_dumping_torque', ankle_s42_source.get_joint_dumping_torque),
        ('is_ankle_pos_legal', ankle_s42_source.is_ankle_pos_legal),
    ]
    
    exported_count = 0
    failed_count = 0
    
    for func_name, func in functions_to_export:
        try:
            output_path = os.path.join(output_dir, f"{func_name}.pt")
            
            # 保存 TorchScript 函数
            torch.jit.save(func, output_path)
            print(f"✓ 导出成功: {func_name}.pt")
            
            # 验证文件能否正常加载
            loaded_func = torch.jit.load(output_path)
            print(f"✓ 验证加载: {func_name}.pt")
            
            exported_count += 1
            
        except Exception as e:
            print(f"✗ 导出失败 {func_name}: {e}")
            failed_count += 1
    
    print(f"\n导出统计:")
    print(f"成功导出: {exported_count} 个函数")
    print(f"导出失败: {failed_count} 个函数")
    print(f"文件保存在: {output_dir}")


def test_exported_functions(output_dir: str):
    """
    测试导出的函数是否工作正常
    
    Args:
        output_dir: 包含 .pt 文件的目录
    """
    print("\n开始测试导出的函数...")
    
    # 创建测试数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    num_envs = 10
    joint_pos = torch.rand(num_envs, 12, device=device)
    joint_vel = torch.rand(num_envs, 12, device=device)
    motor_kd = torch.ones(num_envs, 12, device=device)
    
    # 生成合法的踝关节位置用于测试
    try:
        legal_ankle_pos = ankle_s42_source.generate_diamond_points(num_envs)
        joint_pos[:, 4:6] = legal_ankle_pos[:num_envs].to(device)
        joint_pos[:, 10:12] = legal_ankle_pos[:num_envs].to(device)
        print("✓ 生成合法踝关节位置")
    except Exception as e:
        print(f"⚠ 使用随机踝关节位置: {e}")
        joint_pos[:, 4:6] = 0.0
        joint_pos[:, 10:12] = 0.0
    
    # 测试函数
    test_cases = [
        {
            'name': 'joint_to_motor_position',
            'args': [joint_pos],
            'description': '关节位置转换为电机位置'
        },
        {
            'name': 'is_ankle_pos_legal', 
            'args': [joint_pos[:, 4:6]],
            'description': '检查踝关节位置合法性'
        },
    ]
    
    motor_pos = None
    
    for test_case in test_cases:
        func_name = test_case['name']
        func_args = test_case['args']
        description = test_case['description']
        
        try:
            # 加载函数
            func_path = os.path.join(output_dir, f"{func_name}.pt")
            if not os.path.exists(func_path):
                print(f"⚠ 跳过测试 {func_name}: 文件不存在")
                continue
                
            loaded_func = torch.jit.load(func_path)
            
            # 调用函数
            result = loaded_func(*func_args)
            
            print(f"✓ {func_name}: {description}")
            print(f"  输入形状: {[arg.shape for arg in func_args]}")
            print(f"  输出形状: {result.shape}")
            
            # 保存 motor_pos 用于后续测试
            if func_name == 'joint_to_motor_position':
                motor_pos = result
                
        except Exception as e:
            print(f"✗ {func_name} 测试失败: {e}")
    
    # 测试需要 motor_pos 的函数
    if motor_pos is not None:
        advanced_tests = [
            {
                'name': 'joint_to_motor_velocity',
                'args': [joint_pos, motor_pos, joint_vel],
                'description': '关节速度转换为电机速度'
            },
            {
                'name': 'get_joint_dumping_torque',
                'args': [joint_pos, motor_pos, motor_kd, joint_vel],
                'description': '获取关节阻尼力矩'
            },
            {
                'name': 'motor_to_joint_torque',
                'args': [joint_pos, motor_pos, motor_kd * joint_vel],
                'description': '电机力矩转换为关节力矩'
            },
        ]
        
        for test_case in advanced_tests:
            func_name = test_case['name']
            func_args = test_case['args']
            description = test_case['description']
            
            try:
                func_path = os.path.join(output_dir, f"{func_name}.pt")
                if not os.path.exists(func_path):
                    print(f"⚠ 跳过测试 {func_name}: 文件不存在")
                    continue
                    
                loaded_func = torch.jit.load(func_path)
                result = loaded_func(*func_args)
                
                print(f"✓ {func_name}: {description}")
                print(f"  输入形状: {[arg.shape for arg in func_args]}")
                print(f"  输出形状: {result.shape}")
                
            except Exception as e:
                print(f"✗ {func_name} 测试失败: {e}")
    
    print(f"\n测试完成！")


def main():
    parser = argparse.ArgumentParser(description="导出 ankle_s42_source.py 的 TorchScript 函数")
    parser.add_argument("--output_dir", type=str,
                       default="../ext_template/actuators/torchscript",
                       help="输出目录路径（相对于脚本所在目录）")
    parser.add_argument("--test", action="store_true",
                       help="导出后进行功能测试")
    parser.add_argument("--test_only", action="store_true",
                       help="仅测试现有的 .pt 文件")
    
    args = parser.parse_args()
    
    # 解析输出目录路径
    if not os.path.isabs(args.output_dir):
        # 相对路径，相对于脚本所在目录
        script_dir = Path(__file__).parent
        output_dir = str(script_dir / args.output_dir)
    else:
        output_dir = args.output_dir
    
    print(f"输出目录: {output_dir}")
    
    if not args.test_only:
        # 导出函数
        export_ankle_s42_functions(output_dir)
    
    # 可选测试
    if args.test or args.test_only:
        test_exported_functions(output_dir)
    
    if not args.test_only:
        print(f"\n使用说明:")
        print(f"1. 生成的 .pt 文件已保存在: {output_dir}")
        print(f"2. ankle_s42.py 将自动从该目录加载这些文件")
        print(f"3. 现在可以安全删除 ankle_s42_source.py 文件")


if __name__ == "__main__":
    main()
