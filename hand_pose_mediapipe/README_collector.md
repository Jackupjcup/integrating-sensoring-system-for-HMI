# 手指角度收集器使用说明

## 概述

`FingerAngleCollector` 类是一个用于实时收集手指角度数据和视频的工具。它使用 MediaPipe 进行手部关键点检测，计算手指关节角度，并将数据保存到 `record` 目录中。

## 功能特性

1. **实时手部检测**：使用 MediaPipe 检测手部关键点
2. **角度计算**：计算每个手指关节的弯曲角度
3. **数据记录**：将时间戳和角度数据保存到 CSV 文件
4. **视频录制**：同步录制完整的视频
5. **会话管理**：每个录制会话都有独立的目录和时间戳
6. **线程支持**：可选线程模式以提高性能

## 安装依赖

确保已安装所需依赖：

```bash
pip install mediapipe opencv-python numpy
```

## 快速开始

### 基本用法

```python
from bend_angle_collect import FingerAngleCollector

# 创建收集器实例
collector = FingerAngleCollector(
    record_dir="record",          # 记录目录
    camera_id=0,                  # 摄像头ID
    use_threading=True,           # 使用线程模式
    num_hands=2,                  # 最大手部数量
    model_asset_path='hand_landmarker.task'  # MediaPipe模型文件
)

# 运行收集器
collector.run()
```

### 运行测试脚本

```bash
python dataset_collector.py
```

## 控制说明

程序运行后，可以使用以下键盘控制：

- **按 'r' 键**：开始/停止录制
- **按 'q' 键**：退出程序

## 输出文件结构

每个录制会话会在 `record` 目录下创建独立的子目录：

```
record/
├── session_20250311_143022/     # 会话目录（时间戳格式）
│   ├── angles_data.csv          # 角度数据（CSV格式）
│   ├── video.avi               # 录制的视频
│   └── session_info.json       # 会话信息
└── session_20250311_143125/
    ├── angles_data.csv
    ├── video.avi
    └── session_info.json
```

### 角度数据格式 (angles_data.csv)

CSV文件包含以下列：

| 列名 | 描述 | 示例 |
|------|------|------|
| timestamp | ISO格式时间戳 | 2025-03-11T14:30:22.123456 |
| frame_index | 帧索引 | 0, 1, 2, ... |
| hand_index | 手部索引 | 0 (第一只手), 1 (第二只手) |
| hand_label | 手部标签 | "Left", "Right" |
| thumb_cmc_P_M | 拇指CMC-MCP关节角度 | 120.5 |
| thumb_P_M_I | 拇指MCP-IP关节角度 | 110.2 |
| index_wrist_P_M | 食指腕部-MCP关节角度 | 150.2 |
| index_P_M_I | 食指MCP-PIP关节角度 | 130.1 |
| index_M_I_tip | 食指PIP-DIP关节角度 | 100.5 |
| ... | 其他手指的类似列 | ... |

### 会话信息格式 (session_info.json)

```json
{
  "session_id": "session_20250311_143022",
  "start_time": "2025-03-11T14:30:22.123456",
  "end_time": "2025-03-11T14:30:45.789012",
  "duration_seconds": 23.665556,
  "camera_id": 0,
  "video_codec": "mp4v",
  "video_fps": 30.0,
  "num_hands": 2,
  "fieldnames": ["timestamp", "frame_index", "hand_index", "hand_label", ...]
}
```

## 配置参数

`FingerAngleCollector` 类的初始化参数：

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| record_dir | str | "record" | 记录目录路径 |
| camera_id | int | 0 | 摄像头ID（0通常是默认摄像头） |
| use_threading | bool | True | 是否使用线程模式 |
| num_hands | int | 2 | 检测的最大手部数量 |
| model_asset_path | str | 'hand_landmarker.task' | MediaPipe模型文件路径 |
| min_hand_detection_confidence | float | 0.5 | 手部检测置信度阈值 |
| min_hand_presence_confidence | float | 0.5 | 手部存在置信度阈值 |
| min_tracking_confidence | float | 0.5 | 跟踪置信度阈值 |
| video_codec | str | 'mp4v' | 视频编码器（'mp4v', 'XVID', 'MJPG'等） |
| video_fps | float | 30.0 | 视频帧率 |

## 高级用法

### 手动控制录制

```python
from bend_angle_collect import FingerAngleCollector

collector = FingerAngleCollector()

# 手动开始录制
collector.start_recording()

# ... 执行一些操作 ...

# 手动停止录制
collector.stop_recording()
```

### 自定义数据记录

你可以扩展类来添加自定义数据记录逻辑：

```python
class CustomCollector(FingerAngleCollector):
    def _record_angles(self, timestamp, frame_index, hand_index, hand_label, angles):
        # 调用父类方法记录基本数据
        super()._record_angles(timestamp, frame_index, hand_index, hand_label, angles)

        # 添加自定义记录逻辑
        print(f"记录第{frame_index}帧，手部{hand_label}的角度")
```

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头ID是否正确
   - 确保摄像头没有被其他程序占用

2. **模型文件不存在**
   - MediaPipe会自动下载模型，但需要网络连接
   - 可以手动下载模型文件：https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

3. **视频无法播放**
   - 尝试不同的视频编码器（如'XVID'）
   - 确保有合适的解码器

4. **性能问题**
   - 关闭线程模式：`use_threading=False`
   - 降低视频分辨率

### 错误处理

如果遇到错误，程序会打印错误信息并继续运行。主要错误包括：
- 摄像头错误
- 模型加载错误
- 文件写入错误
- 线程同步错误

## 注意事项

1. **文件大小**：视频文件可能较大，请确保有足够的磁盘空间
2. **性能要求**：实时处理需要一定的CPU/GPU性能
3. **光照条件**：良好的光照条件有助于提高检测精度
4. **手部位置**：保持手部在摄像头视野内，避免遮挡

## 扩展开发

类的主要方法可以重写以扩展功能：

- `_record_angles()`: 自定义角度记录逻辑
- `_record_video_frame()`: 自定义视频帧处理
- `_setup_recording()`: 自定义录制设置
- `_cleanup_recording()`: 自定义清理逻辑

## 许可证

此代码基于原始 `bend_angle_collect.py` 文件开发，保留了原有的函数以保持向后兼容性。