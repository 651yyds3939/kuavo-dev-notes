"""
使用 TorchScript 模型替代原始 ankle_s42.py 的实现
这个模块加载同目录下的预编译 .pt 文件，提供与原始 ankle_s42.py 相同的接口
"""

import torch
import os
from pathlib import Path


class AnkleS42TorchScript:
    """
    加载 TorchScript 模型的封装类，提供与原始 ankle_s42.py 相同的函数接口
    """
    
    def __init__(self, scripts_path: str = None):
        """
        初始化 TorchScript 模型加载器
        
        Args:
            scripts_path: TorchScript 文件所在的目录路径，默认为当前文件所在目录
        """
        if scripts_path is None:
            # 使用当前文件所在目录下的 torchscript 子目录
            scripts_path = Path(__file__).parent / "torchscript"
        
        self.scripts_path = str(scripts_path)
        self._load_functions()
    
    def _load_functions(self):
        """加载所有 TorchScript 函数"""
        try:
            # 加载核心函数
            self._joint_to_motor_position = torch.jit.load(
                os.path.join(self.scripts_path, "joint_to_motor_position.pt")
            )
            self._get_joint_dumping_torque = torch.jit.load(
                os.path.join(self.scripts_path, "get_joint_dumping_torque.pt")
            )
            self._is_ankle_pos_legal = torch.jit.load(
                os.path.join(self.scripts_path, "is_ankle_pos_legal.pt")
            )
            
            # 尝试加载其他可选函数
            try:
                self._joint_to_motor_velocity = torch.jit.load(
                    os.path.join(self.scripts_path, "joint_to_motor_velocity.pt")
                )
            except FileNotFoundError:
                self._joint_to_motor_velocity = None
                
            try:
                self._motor_to_joint_torque = torch.jit.load(
                    os.path.join(self.scripts_path, "motor_to_joint_torque.pt")
                )
            except FileNotFoundError:
                self._motor_to_joint_torque = None
            
            # 尝试加载其他矩阵计算函数
            try:
                self._get_left_ankle_matrix = torch.jit.load(
                    os.path.join(self.scripts_path, "get_left_ankle_matrix.pt")
                )
            except FileNotFoundError:
                self._get_left_ankle_matrix = None
                
            try:
                self._get_right_ankle_matrix = torch.jit.load(
                    os.path.join(self.scripts_path, "get_right_ankle_matrix.pt")
                )
            except FileNotFoundError:
                self._get_right_ankle_matrix = None
                
            try:
                self._get_knee_matrix = torch.jit.load(
                    os.path.join(self.scripts_path, "get_knee_matrix.pt")
                )
            except FileNotFoundError:
                self._get_knee_matrix = None
                
            print(f"✓ 成功加载 TorchScript 模型从: {self.scripts_path}")
            
        except Exception as e:
            raise RuntimeError(f"无法加载 TorchScript 文件从 {self.scripts_path}: {e}")
    
    def joint_to_motor_position(self, q):
        """关节位置转换为电机位置"""
        return self._joint_to_motor_position(q)
    
    def get_joint_dumping_torque(self, q, p, kd, qd):
        """获取关节阻尼力矩"""
        return self._get_joint_dumping_torque(q, p, kd, qd)
    
    def is_ankle_pos_legal(self, points):
        """检查踝关节位置合法性"""
        return self._is_ankle_pos_legal(points)
    
    def joint_to_motor_velocity(self, q, p, dq):
        """关节速度转换为电机速度"""
        if self._joint_to_motor_velocity is None:
            raise RuntimeError("joint_to_motor_velocity.pt 文件未找到")
        return self._joint_to_motor_velocity(q, p, dq)
    
    def motor_to_joint_torque(self, q, p, i):
        """电机力矩转换为关节力矩"""
        if self._motor_to_joint_torque is None:
            raise RuntimeError("motor_to_joint_torque.pt 文件未找到")
        return self._motor_to_joint_torque(q, p, i)
    
    def get_left_ankle_matrix(self, q5, q6, p5, p6):
        """获取左踝关节矩阵"""
        if self._get_left_ankle_matrix is None:
            raise RuntimeError("get_left_ankle_matrix.pt 文件未找到")
        return self._get_left_ankle_matrix(q5, q6, p5, p6)
    
    def get_right_ankle_matrix(self, q11, q12, p11, p12):
        """获取右踝关节矩阵"""
        if self._get_right_ankle_matrix is None:
            raise RuntimeError("get_right_ankle_matrix.pt 文件未找到")
        return self._get_right_ankle_matrix(q11, q12, p11, p12)
    
    def get_knee_matrix(self, q4, p4):
        """获取膝关节矩阵"""
        if self._get_knee_matrix is None:
            raise RuntimeError("get_knee_matrix.pt 文件未找到")
        return self._get_knee_matrix(q4, p4)


# 全局实例，延迟初始化
_ankle_s42_instance = None


def _get_instance(scripts_path: str = None):
    """获取全局 TorchScript 实例"""
    global _ankle_s42_instance
    if _ankle_s42_instance is None:
        _ankle_s42_instance = AnkleS42TorchScript(scripts_path)
    return _ankle_s42_instance


# 提供与原始 ankle_s42.py 相同的函数接口
def joint_to_motor_position(q):
    """关节位置转换为电机位置"""
    return _get_instance().joint_to_motor_position(q)


def get_joint_dumping_torque(q, p, kd, qd):
    """获取关节阻尼力矩"""
    return _get_instance().get_joint_dumping_torque(q, p, kd, qd)


def is_ankle_pos_legal(points):
    """检查踝关节位置合法性"""
    return _get_instance().is_ankle_pos_legal(points)


def joint_to_motor_velocity(q, p, dq):
    """关节速度转换为电机速度"""
    return _get_instance().joint_to_motor_velocity(q, p, dq)


def motor_to_joint_torque(q, p, i):
    """电机力矩转换为关节力矩"""
    return _get_instance().motor_to_joint_torque(q, p, i)


def get_left_ankle_matrix(q5, q6, p5, p6):
    """获取左踝关节矩阵"""
    return _get_instance().get_left_ankle_matrix(q5, q6, p5, p6)


def get_right_ankle_matrix(q11, q12, p11, p12):
    """获取右踝关节矩阵"""
    return _get_instance().get_right_ankle_matrix(q11, q12, p11, p12)


def get_knee_matrix(q4, p4):
    """获取膝关节矩阵"""
    return _get_instance().get_knee_matrix(q4, p4)


# 兼容性函数，用于指定脚本路径
def initialize_torchscript(scripts_path: str):
    """
    初始化 TorchScript 模型路径
    
    Args:
        scripts_path: TorchScript 文件所在的目录路径
    """
    global _ankle_s42_instance
    _ankle_s42_instance = AnkleS42TorchScript(scripts_path)


# 保持与原始文件的兼容性
def generate_diamond_points(num_points=8092, ratio=10):
    """
    生成菱形区域内的合法踝关节位置点
    这个函数保持原始实现，因为它主要用于测试和数据生成
    """
    vertices = torch.tensor([
        [0.34, 0.],
        [-0.26, 0.8],
        [-0.87, 0.],
        [-0.26, -0.8],
    ])
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    points = torch.rand(num_points * ratio, 2) * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor([min_x, min_y])
    if_in = is_ankle_pos_legal(points)
    assert torch.sum(if_in) >= num_points
    return points[if_in][:num_points]


if __name__ == '__main__':
    # 测试模块
    print("测试 ankle_s42 TorchScript 模块...")
    
    # 创建测试数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_q = torch.rand(10, 12, device=device)
    test_kd = torch.ones(10, 12, device=device)
    test_qd = torch.rand(10, 12, device=device)
    
    try:
        # 测试函数调用
        motor_pos = joint_to_motor_position(test_q)
        dumping_torque = get_joint_dumping_torque(test_q, motor_pos, test_kd, test_qd)
        ankle_legal = is_ankle_pos_legal(test_q[:, 4:6])
        
        print("✓ 所有函数测试通过")
        print(f"输入数据形状: {test_q.shape}")
        print(f"电机位置形状: {motor_pos.shape}")
        print(f"阻尼力矩形状: {dumping_torque.shape}")
        print(f"踝关节合法性: {ankle_legal.shape}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("请确保同目录下有相应的 .pt 文件")
