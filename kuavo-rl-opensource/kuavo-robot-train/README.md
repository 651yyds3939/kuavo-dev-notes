# kuavo-robot-train — Gym 版训练（可选）

> 本仓库子目录；主部署请用 [`kuavo-robot-deploy`](../kuavo-robot-deploy/readme.md)。Lab 训练见 [`leju_robot_rl`](../../../leju_robot_rl/README.md)。

---

## isaacgym
```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4  #将名字改为 isaacgym 并存放在'kuavo-robot-train'同级目录下
```
## conda
推荐使用MiniConda，轻量化使用更灵活。
```bash
#安装
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
#初始化
~/miniconda3/bin/conda init --all
source ~/.bashrc
```
## Environment
```bash
conda create -n humanoid-gym-op python=3.8
conda activate humanoid-gym-op
cd kuavo-rl-opensource/kuavo-robot-train
pip install -e ../isaacgym/python
pip install -e .
```
# Usage Guide
## Training and Playing
### Examples
```bash
python scripts/train.py --task=kuavo_s42_sk_ppo --run_name v1 --headless --num_envs 4096
python scripts/play.py --task=kuavo_s42_sk_ppo --run_name v1
```
## Resuming Training and Play Specified '.pt' Version
### Examples
```bash
python scripts/train.py --task=kuavo_s42_sk_ppo --run_name v1 --headless --num_envs 4096 --resume --load_run Feb07_11-27-10_v1 --checkpoint 0 #该例子表示文件保存在'kuavo-robot-train/logs/Kuavo_s42_sk_ppo/Feb07_11-27-10_v1'目录下，选择了model_0.pt文件进行继续训练，headless选项关闭了GUI显示。
python scripts/play.py --task=kuavo_s42_sk_ppo --run_name v1 --load_run Feb07_11-27-10_v1 --checkpoint 0
```

## 参数说明
### 自定义参数详解

#### 1. 训练任务控制
| 参数名 | 类型 | 默认值 | 关键作用 | 使用场景示例 |
|--------|------|--------|----------|--------------|
| `--task` | str | `XBotL_free` | 定义训练任务类型 | 切换机器人型号（如`XBotL_walking`）|
| `--max_iterations` | int | - | 最大训练迭代次数 | 限制训练时长 |
| `--resume` | flag | False | 从检查点恢复训练 | 训练意外中断后继续 |

#### 2. 实验版本管理
| 参数名 | 类型 | 关键作用 | 数据存储逻辑 |
|--------|------|----------|--------------|
| `--experiment_name` | str | 标识实验项目 | 对应 `experiments/` 下的文件夹 |
| `--run_name` | str | 单次运行标识 | 同一实验的不同参数对比 |
| `--load_run` | str | 加载历史运行记录 | `-1` 表示加载最新运行 |
| `--checkpoint` | int | 指定模型检查点 | `-1` 加载最新检查点 |

#### 3. 硬件资源配置
| 参数名 | 类型 | 默认值 | 技术细节 | 典型配置 |
|--------|------|--------|----------|----------|
| `--rl_device` | str | `cuda:0` | RL算法运行的设备 | `cuda:1` 指定第二块GPU |
| `--num_envs` | int | - | 并行环境数 | 4096（需根据显存调整）|
| `--horovod` | flag | False | 启用分布式训练 | 多GPU训练时使用 |

#### 4. 可视化与调试
| 参数名 | 类型 | 关键作用 | 典型使用场景 |
|--------|------|----------|--------------|
| `--headless` | flag | 禁用图形渲染 | 服务器无显示器环境 |
| `--seed` | int | 固定随机种子 | 实验可复现性保障 |

#### 5. 参数优先级逻辑
  - 命令行参数 > 配置文件 > 代码默认值