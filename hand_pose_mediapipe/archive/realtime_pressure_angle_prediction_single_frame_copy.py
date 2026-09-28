"""
基于PressureHeat_RowColsLines.py修改的实时压力-角度预测可视化
左边：压力热图
右边：模型预测的14个角度值
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import torch
import sys
import os

# 导入自定义模型和数据处理类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smallCNN_regression import SmallCNN_regression
from train_cnn_regression import PressureAngleDataset

# 导入传感器相关模块
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading
from assamble_dataset import raw_data_to_true_img

# 常量设置
ROWS, COLS = 14, 10
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100
MAX_VALUE = 500
MIN_VALUE = -100

initial_data = np.zeros(TOTAL_NUM)

# 加载配置
imu_config = load_config("/IMUProcessing/config.json")
palm_points_config = load_config("/PalmPoints_and_FingerBend/PalmPointsConfig.json")

# 配置串口
ser = serial.Serial(
    port=imu_config['serial']['port'],
    baudrate=imu_config['serial']['baud_rate'],
    timeout=imu_config['serial']['timeout']
)

# 初始化IMU参数
imu_config_path = "/IMUProcessing/config.json"
imu_1 = IMUData('mpu6050_1', imu_config_path)
imu_2 = IMUData('mpu6050_2', imu_config_path)
imu_3 = IMUData('mpu6050_3', imu_config_path)

# 初始化Palm Points
palm_points = PalmPointsClass(ROWS, COLS, "/PalmPoints_and_FingerBend/PalmPointsConfig.json")

# 初始化数据收集线程
threading_collect = ThreadingCollect(ser=ser,
                                     imu_1=imu_1,
                                     imu_2=imu_2,
                                     imu_3=imu_3,
                                     palm_points=palm_points,
                                     duration=500)
with open("../train/evaluation_report.json", "r", encoding='utf-8') as f:
    evaluation_results = json.load(f)

pressure_mean, pressure_std = evaluation_results["pressure_mean"], evaluation_results["pressure_std"]

# ---------------------- 1. 创建画布和布局 ----------------------
fig = plt.figure(figsize=(12, 6))  # 设置画布大小

# 使用gridspec划分布局：1行2列
# 左侧大图占1列，右侧角度显示占1列
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

# 创建两个轴对象
ax1 = fig.add_subplot(gs[0, 0])  # 左侧压力热图
ax2 = fig.add_subplot(gs[0, 1])  # 右侧角度显示

# 隐藏右侧轴的坐标轴
ax2.axis('off')

# ---------------------- 2. 初始化图像对象 ----------------------
# 创建压力热图初始图像对象
scatter = ax1.scatter(
    [coord[0] for coord in palm_points.plane_pressure_points_coords],  # 所有点的x坐标
    [-coord[1] for coord in palm_points.plane_pressure_points_coords],
    c=initial_data,                         # 每个点的压力值（颜色）
    cmap='coolwarm',
    vmin=MIN_VALUE,
    vmax=MAX_VALUE,
    marker='s',
    s=200,                                  # 点的大小
    edgecolors='white',                     # 点的边框
    linewidth=0.5,
)
# 添加侧边颜色条
plt.colorbar(scatter, ax=ax1, label='Pressure Intensity')

# 添加压力值文本标签
texts = []
for sensor_coord in palm_points.plane_pressure_points_coords:
    x, y = sensor_coord
    text = ax1.text(x, -y, "", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)

# 初始化角度显示文本
angle_texts = []
# 手指关节角度标签: 拇指(2) + 食指(3) + 中指(3) + 无名指(3) + 小指(3) = 14个角度
angle_labels = [
    # 拇指 (2个角度)
    "P", "M",
    # 食指 (3个角度)
    "index_P", "index_M", "index_I",
    # 中指 (3个角度)
    "middle_P", "middle_M", "middle_I",
    # 无名指 (3个角度)
    "ring_P", "ring_M", "ring_I",
    # 小指 (3个角度)
    "pinky_P", "pinky_M", "pinky_I"
]

# 在右侧创建角度显示
y_positions = np.linspace(0.9, 0.1, 14)  # 14个角度值从上到下均匀分布
for i, (label, y_pos) in enumerate(zip(angle_labels, y_positions)):
    # 标签
    label_text = ax2.text(0.08, y_pos, f"{label}:", fontsize=9,
                         verticalalignment='center', transform=ax2.transAxes)
    # 值（初始为0）
    value_text = ax2.text(0.65, y_pos, "0.00°", fontsize=12, fontweight='bold',
                         verticalalignment='center', transform=ax2.transAxes,
                         color='red')
    angle_texts.append((label_text, value_text))

# 添加标题
ax2.text(0.5, 0.95, "Predicted Hand Joint Angles", fontsize=14, fontweight='bold',
        horizontalalignment='center', transform=ax2.transAxes)

# ---------------------- 3. 加载训练好的模型 ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载模型
model = SmallCNN_regression(num_classes=14).to(device)
model_path = '../train/best_cnn_regression_model.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]+1}')
    print(f'Model validation loss: {checkpoint["val_loss"]:.6f}')
else:
    print(f'Warning: Model file {model_path} not found!')
    print('Using untrained model...')

model.eval()  # 设置为评估模式

# 加载标准化参数（从训练集中获取）
try:
    # 尝试加载训练时保存的标准化参数
    results_path = '../train/training_results.npy'
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        print('Loaded normalization parameters from training results')
    else:
        # 如果没有保存的参数，使用默认值
        pressure_mean = 0.0
        pressure_std = 1.0
        print('Using default normalization parameters')
except:
    pressure_mean = 0.0
    pressure_std = 1.0
    print('Failed to load normalization parameters, using defaults')

data_lock = threading.Lock()

# ---------------------- 4. 定义动画更新函数 ----------------------
def update(frame):
    """
    动画更新函数，每次帧刷新时调用
    """
    with data_lock:
        # 获取最新压力数据
        if hasattr(threading_collect, 'palm_points') and threading_collect.palm_points.latest_pressure_data is not None:
            raw_data = threading_collect.palm_points.latest_pressure_data
        else:
            raw_data = np.zeros(TOTAL_NUM)  # 无数据时显示0

    # 更新压力热图
    scatter.set_array(raw_data)

    # 更新压力值文本标签
    for idx in range(TOTAL_NUM):
        val = raw_data[idx]
        texts[idx].set_text(f"{int(raw_data[idx])}")

        # 智能调整文字颜色
        if val > MAX_VALUE * 0.6:
            texts[idx].set_color("white")
        else:
            texts[idx].set_color("black")

    # 使用模型进行角度预测
    with torch.no_grad():
        # 预处理压力数据
        if np.sum(np.abs(raw_data)) > 0:  # 如果有有效数据
            # 重塑为14x10
            raw_data = raw_data_to_true_img(raw_data.reshape(ROWS, COLS))
            pressure_data = raw_data.astype(np.float32)

            # 应用标准化（这里需要加载训练时的标准化参数）
            # 暂时跳过标准化，因为需要训练集的均值和标准差
            pressure_data = (pressure_data - pressure_mean) / (pressure_std + 1e-6)

            # 转换为torch张量
            pressure_tensor = torch.from_numpy(pressure_data).unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 10)
            pressure_tensor = pressure_tensor.to(device)

            # 模型预测
            predicted_angles = model(pressure_tensor)
            predicted_angles_np = predicted_angles.cpu().numpy().flatten()
            predicted_angles_np = np.clip(predicted_angles_np, a_min=0, a_max=1) * 180
        else:
            predicted_angles_np = np.zeros(14)

    # 更新角度显示
    for i, (label_text, value_text) in enumerate(angle_texts):
        angle_value = predicted_angles_np[i]
        value_text.set_text(f"{angle_value:.2f}°")

        # 根据角度值设置颜色
        if angle_value > 170:
            value_text.set_color('green')  # 接近伸直
        elif angle_value > 100:
            value_text.set_color('orange')  # 中等弯曲
        else:
            value_text.set_color('red')  # 严重弯曲

    # 返回更新的对象
    return scatter, *texts, *[t for pair in angle_texts for t in pair]

# ---------------------- 5. 启动数据收集和动画 ----------------------
threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()

print('Starting data collection thread...')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

print('Starting visualization...')

# 创建动画对象
ani = FuncAnimation(
    fig=fig,
    func=update,
    frames=np.arange(0, 1000),  # 更多帧数
    interval=UPDATE_INTERVAL,   # 更新间隔
    blit=True,
    repeat=True
)

# 显示画布
plt.tight_layout()
plt.show()

print('Visualization finished.')