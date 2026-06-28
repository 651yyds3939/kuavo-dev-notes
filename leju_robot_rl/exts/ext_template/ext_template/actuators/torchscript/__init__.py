"""TorchScript模型文件包。

此包包含预编译的TorchScript模型文件(.pt)，用于机器人控制相关的计算。
"""

import os
from pathlib import Path

# 获取当前包的路径
TORCHSCRIPT_DIR = Path(__file__).parent

def get_model_path(model_name: str) -> Path:
    """
    获取指定模型文件的完整路径。
    
    Args:
        model_name: 模型文件名（包含.pt扩展名）
        
    Returns:
        模型文件的完整路径
        
    Raises:
        FileNotFoundError: 如果指定的模型文件不存在
    """
    model_path = TORCHSCRIPT_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_name}' not found in {TORCHSCRIPT_DIR}")
    return model_path

def list_available_models() -> list[str]:
    """
    列出所有可用的模型文件。
    
    Returns:
        可用模型文件名的列表
    """
    return [f.name for f in TORCHSCRIPT_DIR.glob("*.pt")]
