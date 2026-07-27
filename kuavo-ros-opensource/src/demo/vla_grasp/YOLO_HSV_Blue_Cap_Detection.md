# YOLO + HSV 蓝色瓶盖检测方案

## 一、背景与动机

### 1.1 原有方案

机器人使用 YOLOv8 检测瓶身（bottle 类别），取检测框中心坐标，通过深度图投影到相机 3D 坐标系，再经 TF2 变换到 `base_link` 坐标系，发布到 `/vla/yolo_target`（`geometry_msgs/PointStamped`）。

### 1.2 原有方案的问题

下位机（NUC）基于 YOLO 瓶身中心坐标，用几何推算瓶盖位置。由于瓶身检测框只能给出瓶子大致中心，瓶盖在瓶子顶端，几何推算误差太大，抓取成功率低。

### 1.3 改进目标

在原有 YOLO 瓶身检测的基础上，增加一个独立的 HSV 颜色过滤通道，直接从图像中检测**深蓝色瓶盖**（百事可乐瓶盖），精确给出瓶盖在 `base_link` 坐标系下的 3D 坐标，发布到新话题 `/vla/cap_target`。

---

## 二、涉及文件

| 文件 | 说明 |
|------|------|
| `kuavo_ros_application/src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py` | 修改的唯一文件，YOLO 推理 + HSV 瓶盖检测 |

---

## 三、实现原理

### 3.1 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                    process_frame()                        │
│                                                          │
│  输入: RGB 图像 + 深度图 + CameraInfo                     │
│                                                          │
│  ┌─────────────────┐   ┌──────────────────────────┐     │
│  │  YOLOv8 推理     │   │  HSV 蓝色瓶盖检测 (新增)  │     │
│  │  ↓              │   │  ↓                       │     │
│  │  检测 bottle    │   │  YOLO 瓶子框上 35% → ROI  │     │
│  │  ↓              │   │  ↓                       │     │
│  │  框中心 → 3D    │   │  BGR→HSV→inRange→轮廓    │     │
│  │  ↓              │   │  ↓                       │     │
│  │  TF → base_link │   │  质心 → 深度 → 3D        │     │
│  │  ↓              │   │  ↓                       │     │
│  │  /vla/yolo_target│  │  TF → base_link           │     │
│  └─────────────────┘   │  ↓                       │     │
│                         │  /vla/cap_target          │     │
│                         └──────────────────────────┘     │
│                                                          │
│  输出: combined_img (标注画面) + Detection2DArray        │
└──────────────────────────────────────────────────────────┘
```

### 3.2 HSV 颜色过滤原理

1. **颜色空间转换**：BGR → HSV，OpenCV 的 H 通道范围 0-179
2. **蓝色阈值**：`H:[100, 130], S:[50, 255], V:[50, 255]`
   - H 100-130 覆盖深蓝到蓝紫区间，对应百事可乐标志性深蓝色
   - S 50+ 过滤低饱和度灰白区域
   - V 50+ 过滤过暗的阴影
3. **形态学滤波**：3×3 椭圆核开运算 + 闭运算，去除椒盐噪声
4. **轮廓提取**：`cv2.findContours` → 取最大轮廓 → 计算图像矩求质心
5. **深度采样**：以质心为中心取 10×10 窗口，中值滤波去噪
6. **3D 投影**：使用相机内参 K 矩阵，将像素坐标转为相机坐标系 3D 点
7. **TF2 变换**：查询 `base_link` ← `camera_frame` 的实时变换，抵消头部晃动

### 3.3 YOLO 引导 ROI 搜索（核心创新）

不搜索全图，而是利用 YOLO 已检测到的瓶子框来限定 HSV 搜索范围：

```
YOLO 瓶子框 [bx1, by1, bx2, by2]
  → 取框的 35% 高度（瓶盖所在端）
  → 水平方向左右各扩展 10%
  → 作为 HSV 搜索 ROI
```

**ROI 内的优势**：
- 面积阈值从 100px 降至 20px（ROI 模式下自动 ÷5），远距离小瓶盖也能检测
- 彻底排除 ROI 外的蓝色干扰（地面、衣物等）
- HSV 只处理全图 ~5% 的像素，几乎零性能开销

### 3.4 两种搜索模式

| 模式 | 触发条件 | 适用场景 |
|------|----------|----------|
| **ROI 模式** | YOLO 检测到瓶子 | 瓶盖在框内，高精度 |
| **全图回退** | YOLO 未检测到瓶子 | 兜底，保持独立可用 |

---

## 四、代码改动详情

### 4.1 新增配置常量（L44-54）

```python
# ============================================================
# 🔵 蓝色瓶盖检测开关 (默认关闭，通过 ROS 参数 ~enable_blue_cap_detection 打开)
# ============================================================
ENABLE_BLUE_CAP_DETECTION = False

# 蓝色 HSV 阈值 (百事可乐瓶盖深蓝色)
BLUE_LOWER = np.array([100, 50, 50])   # H:100-130, S:50-255, V:50-255
BLUE_UPPER = np.array([130, 255, 255])
BLUE_MASK_MIN_AREA = 100               # mask 最小面积 (px)，低于此值不发
BLUE_MAX_DEPTH_M = 1.5                # 最大深度 (米)，过滤远处地面等干扰
CAMERA_UPSIDE_DOWN = True              # 相机倒装：设为 True 自动反转 ROI 方向
```

### 4.2 全局变量（L56-57）

```python
vla_pub, cap_pub, tf_buffer = None, None, None  # 新增 cap_pub
```

### 4.3 新增 `detect_blue_cap()` 函数（L92-202）

完整实现 HSV 蓝色检测管线，支持可选 `roi` 参数：

```python
def detect_blue_cap(input_image, depth_image, camera_info, roi=None):
    """
    🔵 HSV 颜色过滤检测蓝色瓶盖
    roi: (x1, y1, x2, y2) 可选，YOLO 瓶身框的上半部分，限定搜索范围
    返回 (PointStamped, cx, cy, contour) 或 None
    """
    # 1. 根据 roi 裁剪搜索区域
    # 2. BGR → HSV → inRange 生成蓝色 mask
    # 3. 形态学开闭运算去噪
    # 4. 面积过滤（ROI 模式下阈值自动降为 1/5）
    # 5. 找最大轮廓 → 图像矩求质心
    # 6. 深度窗口中值采样 → 3D 投影
    # 7. 深度距离过滤（> BLUE_MAX_DEPTH_M 丢弃）
    # 8. TF2 变换 camera → base_link
    # 9. 返回 (PointStamped, cx, cy, contour)
```

### 4.4 `process_frame()` 改动（L213，L221-224，L257-309）

**追踪最佳瓶子框**（L213，L221-224）：

```python
best_bottle_box, best_bottle_box_score = None, 0.0  # 新增

# 在 YOLO 检测循环中：
if best_bottle_box is None or score > best_bottle_box_score:
    best_bottle_box_score = score
    best_bottle_box = (x1, y1, x2, y2)
```

**计算 ROI + 调用瓶盖检测**（L257-309）：

```python
if ENABLE_BLUE_CAP_DETECTION:
    # 用 YOLO 瓶子框计算 ROI
    cap_roi = None
    if best_bottle_box is not None:
        # 相机倒装时取框下 35%，正常取框上 35%
        if CAMERA_UPSIDE_DOWN:
            cap_roi = (bx1 - margin_x, by2 - int(bh * 0.35), bx2 + margin_x, by2)
        else:
            cap_roi = (bx1 - margin_x, by1, bx2 + margin_x, by1 + int(bh * 0.35))

    result = detect_blue_cap(input_image, depth_image, camera_info, roi=cap_roi)
    if result is not None and cap_pub is not None:
        cap_msg, cx, cy, contour = result
        cap_pub.publish(cap_msg)
        # 可视化：蓝色轮廓 + 半透明填充 + 红色十字质心
```

### 4.5 `main()` 改动（L330-339，L348）

**ROS 参数读取**：

```python
ENABLE_BLUE_CAP_DETECTION = rospy.get_param('~enable_blue_cap_detection', False)
```

**新增发布器**：

```python
cap_pub = rospy.Publisher('/vla/cap_target', PointStamped, queue_size=1)
```

---

## 五、使用方式

### 5.1 启动命令

```bash
# 终端 1: 启动相机
roslaunch dynamic_biped load_robot_head.launch

# 终端 2: 启动 YOLO + 瓶盖检测（默认关闭瓶盖检测）
python3 src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py

# 终端 2 变体: 开启瓶盖检测
python3 src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py _enable_blue_cap_detection:=true

# 终端 3: 查看瓶盖坐标
rostopic echo /vla/cap_target

# 终端 4: 查看标注画面
rosrun rqt_image_view rqt_image_view /object_yolo_box_segment_image
```

### 5.2 话题一览

| 话题 | 类型 | 说明 |
|------|------|------|
| `/vla/yolo_target` | `PointStamped` | YOLO 瓶身中心坐标（base_link，原有） |
| `/vla/cap_target` | `PointStamped` | HSV 蓝色瓶盖坐标（base_link，新增） |
| `/object_yolo_box_segment_image` | `Image` | 标注后画面（绿色瓶身框 + 黄色 ROI + 蓝色瓶盖轮廓） |

### 5.3 rqt 画面标注

| 标注 | 颜色 | 形状 | 说明 |
|------|------|------|------|
| 瓶身检测框 | 🟢 绿色 | 矩形框 | YOLO 检测到的 bottle |
| ROI 搜索区域 | 🟡 黄色 | 矩形框 | HSV 搜索范围（瓶身框 35% 高度） |
| 蓝色检测轮廓 | 🔵 蓝色 | 轮廓线 + 半透明填充 | HSV 提取到的蓝色像素区域 |
| 瓶盖质心 | 🔴 红色 | 十字 + 圆点 | 瓶盖精确位置，标注像素坐标 |

---

## 六、遇到的问题与解决方案

### 问题 1：地面蓝色干扰

**现象**：机器人前方地面是蓝色的，HSV 检测总是选中地面（面积最大），3D 坐标飞到远处（X≈2.45m vs 瓶身 X≈0.42m）。

**根因**：`max(contours, key=cv2.contourArea)` 在全图中取最大蓝色轮廓，地面蓝色面积远大于瓶盖。

**解决方案**：
1. 增加深度过滤 `BLUE_MAX_DEPTH_M = 1.5`，远处地面（Z>1.5m）直接丢弃
2. 后期进一步通过 YOLO ROI 机制，地面蓝色根本不在搜索区域内

---

### 问题 2：瓶盖放桌上检测不到，只有贴近相机才能检测

**现象**：瓶盖离相机近（Z≈0.2-0.5m）时能检测，放桌上（Z≈0.7-1.0m）就检测不到。终端只有 YOLO 日志，没有瓶盖日志。

**根因**：
1. 距离远了，瓶盖在图像中像素面积变小，低于 `BLUE_MASK_MIN_AREA = 100` 阈值
2. 距离远了，蓝色饱和度下降、亮度降低，HSV 匹配度下降
3. 深度过滤虽然排除了地面，但瓶盖本身的蓝色区域太小被过滤掉了

**解决方案**：**YOLO 引导 ROI 搜索**
- 利用 YOLO 瓶身检测框的上 35%（瓶盖所在端）作为 HSV 搜索区域
- ROI 模式下面积阈值自动降为 `BLUE_MASK_MIN_AREA // 5 = 20`，远处小瓶盖也能检测
- 搜索范围缩小到全图的 ~5%，其他蓝色物体自然被排除

---

### 问题 3：相机倒装导致 ROI 定位错误

**现象**：
- rqt 画面中文字（FPS 等）是倒的
- 黄色 ROI 框出现在瓶子底部而不是瓶盖端
- 蓝色十字在透明瓶身和桌面之间反复横跳，定位不稳定

**根因**：相机物理倒装在机器人头部，驱动输出原始 sensor 数据不做旋转。YOLO 检测框的"上方"（像素 y 坐标小的一端）在真实世界中是瓶子的底部。

**解决方案**：
- 添加 `CAMERA_UPSIDE_DOWN = True` 配置
- ROI 计算时自动取框的**像素下方**（y 坐标大的一端，即真实世界的瓶盖端）：`by2 - int(bh * 0.35)` 到 `by2`
- 不翻转图像（翻转会破坏深度图和 RGB 的像素对齐以及相机内参对应关系）
- rqt 画面文字保持倒置，但功能完全正常

**为什么不翻转图像**：翻转图像（`cv2.flip(img, -1)`）后，相机内参 K 矩阵不再适用——像素 (u,v) 经过 flip 后对应的是原图 (W-u, H-v) 的世界点，但 K 矩阵仍然是原图的标定值，导致 3D 投影结果错误。保持图像不翻转、只调整 ROI 方向是最安全且零开销的方案。

---

### 问题 4：`cv2.LINE_DASH` 属性不存在

**现象**：运行时抛出 `AttributeError: module 'cv2' has no attribute 'LINE_DASH'`。

**根因**：机器人环境中的 OpenCV 版本较老（OpenCV 4.0 之前），不支持 `cv2.LINE_DASH` 线型。

**解决方案**：去掉 `cv2.LINE_DASH` 参数，使用默认实线绘制 ROI 框。

---

## 七、配置参数速查表

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `ENABLE_BLUE_CAP_DETECTION` | `False` | 瓶盖检测总开关 | ROS 参数 `_enable_blue_cap_detection:=true` 开启 |
| `BLUE_LOWER` | `[100, 50, 50]` | HSV 蓝色下限 | H 调色相（蓝=100-130），S 调饱和度，V 调亮度 |
| `BLUE_UPPER` | `[130, 255, 255]` | HSV 蓝色上限 | 光线暗时降 S/V 下限到 30 |
| `BLUE_MASK_MIN_AREA` | `100` | 全图模式最小面积 (px) | ROI 模式自动降为 1/5 |
| `BLUE_MAX_DEPTH_M` | `1.5` | 最大有效深度 (m) | 桌子远可调大到 2.0 |
| `CAMERA_UPSIDE_DOWN` | `True` | 相机是否倒装 | 正常相机设为 `False` |

---

## 八、典型运行日志

```
[INFO] 🔵 蓝色瓶盖 HSV 检测已启用
[INFO] 🎯 绝对坐标 (抗晃动): X=0.424, Y=-0.078
[INFO] 🔵 瓶盖坐标: X=0.406, Y=0.021, Z=0.504 | 像素:(739,576)
[INFO] 🎯 绝对坐标 (抗晃动): X=0.423, Y=-0.078
[INFO] 🔵 瓶盖坐标: X=0.397, Y=0.024, Z=0.496 | 像素:(744,554)
```

- `🎯` = YOLO 瓶身 → `/vla/yolo_target`
- `🔵` = HSV 瓶盖 → `/vla/cap_target`
- 像素坐标对应 rqt 画面上的红色十字位置

---

## 九、与原方案的关系

| 项目 | 原方案 | 新方案 |
|------|--------|--------|
| YOLO 瓶身检测 | ✅ 不变 | ✅ 不变 |
| `/vla/yolo_target` 发布 | ✅ 不变 | ✅ 不变 |
| HSV 瓶盖检测 | ❌ 无 | ✅ 新增 |
| `/vla/cap_target` 发布 | ❌ 无 | ✅ 新增 |
| 帧率 | 原有帧率 | 不变（HSV 仅处理 ROI 区域，几乎零开销） |
| 两个通道的关系 | - | 完全独立并行，互不影响 |
| 下位机使用建议 | 瓶身中心坐标 | 优先使用 `/vla/cap_target`，更精确 |
