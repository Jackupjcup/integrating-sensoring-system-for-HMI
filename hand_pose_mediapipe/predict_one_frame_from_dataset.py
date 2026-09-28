"""
从dataset.csv中取出某一帧的压力传感器数据进行预测，
使用scatter形式显示压力图，右边显示预测角度和真实角度对比。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("D:/blender_workplace/modules")
from smallCNN_regression import SmallCNN_regression
from IMUProcessing.MPU6050_utils import load_config

# 常量设置
ROWS, COLS = 14, 10
TOTAL_NUM = ROWS * COLS
NUM_FRAMES = 5  # 输入需要5帧

# 加载配置获取传感器坐标
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

# 行转换索引（用于将CNN输入格式转换回真实显示格式）
row_idx_list = [0, 13, 1, 12, 2, 11, 3, 10, 4, 9, 5, 8, 6, 7]

MIN_VALUE = -3
MAX_VALUE = 3


def cnn_img_to_raw(row_idx_list, pressure_array, max_row=13):
    """
    将CNN输入格式的压力数据转换回真实显示格式
    """
    pressure_real = np.zeros(pressure_array.shape)
    for i, row in enumerate(row_idx_list):
        pressure_real[row] = pressure_array[max_row - i]
    return pressure_real


def load_data_from_dataset(csv_path):
    """
    解析dataset.csv中的pressure和angles
    """
    data = pd.read_csv(csv_path)
    pressures = []
    angles = []

    print("Parsing dataset...")
    for idx in range(len(data)):
        # 解析pressure
        pressure_str = data.iloc[idx]['pressure']
        pressure_str = pressure_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
        pressure_array = np.fromstring(pressure_str, sep=' ', dtype=np.float32)
        pressure_array = pressure_array.reshape(ROWS, COLS)

        # 解析angles
        angles_str = data.iloc[idx]['angles']
        angles_str = angles_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
        angles_array = np.fromstring(angles_str, sep=' ', dtype=np.float32)

        pressures.append(pressure_array)
        angles.append(angles_array)

    pressures = np.array(pressures)
    angles = np.array(angles)

    print(f"Dataset loaded: {pressures.shape} pressures, {angles.shape} angles")
    return pressures, angles


def predict_and_visualize(frame_idx=20):
    """
    从指定帧开始，取5帧压力数据预测角度，并可视化对比真实角度
    """
    # 数据路径
    dataset_path = "D:/blender_workplace/modules/hand_pose_mediapipe/record/session_20260316_133259/dataset.csv"
    # model_path = "D:/blender_workplace/modules/hand_pose_mediapipe/train/best_cnn_regression_model.pth"
    model_path = "D:/blender_workplace/modules/hand_pose_mediapipe/train/Archive/Small_CNN_Multi_Regressor_5Frames/best_cnn_regression_model.pth"
    eval_report_path = "D:/blender_workplace/modules/hand_pose_mediapipe/train/evaluation_report.json"

    # 加载标准化参数
    with open(eval_report_path, "r", encoding='utf-8') as f:
        eval_report = json.load(f)
    pressure_mean = eval_report["pressure_mean"]
    pressure_std = eval_report["pressure_std"]

    # 加载数据集
    pressures, angles = load_data_from_dataset(dataset_path)

    # 检查帧索引是否有效（需要连续5帧）
    max_frame_idx = len(pressures) - NUM_FRAMES
    if frame_idx > max_frame_idx:
        frame_idx = max_frame_idx
        print(f"Frame index adjusted to {frame_idx} (max available for 5 frames)")

    # 取连续5帧压力数据
    pressure_frames = pressures[frame_idx:frame_idx + NUM_FRAMES]  # (5, 14, 10)

    # 真实角度（第5帧对应的角度）
    true_angles = angles[frame_idx + NUM_FRAMES - 1]  # (14,)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = SmallCNN_regression(num_classes=14).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from epoch {checkpoint["epoch"]+1}')
    else:
        print(f'Warning: Model file {model_path} not found!')
        return

    model.eval()

    # 标准化压力数据
    pressure_normalized = (pressure_frames - pressure_mean) / (pressure_std + 1e-8)

    # 转换为tensor: (1, 5, 14, 10)
    input_tensor = torch.from_numpy(pressure_normalized.astype(np.float32))
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        predicted_angles = model(input_tensor)
    predicted_angles_np = predicted_angles.cpu().numpy().flatten()

    # 角度值转换到0-180度范围
    predicted_angles_deg = predicted_angles_np * 180
    true_angles_deg = true_angles * 180

    # 计算误差
    errors = np.abs(predicted_angles_deg - true_angles_deg)

    # ========== 可视化 ==========
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0, 0])  # 左侧压力scatter图
    ax2 = fig.add_subplot(gs[0, 1])  # 右侧角度对比
    ax2.axis('off')

    # ----- 左侧：压力scatter图 -----
    # 使用最后一帧的压力数据进行显示
    pressure_last_frame = pressure_frames[NUM_FRAMES - 1]  # (14, 10)

    # 转换为真实显示格式
    pressure_display = cnn_img_to_raw(row_idx_list, pressure_last_frame)
    pressure_display_flat = pressure_display.reshape(140, -1)

    scatter = ax1.scatter(
        [coord[0] for coord in sensor_coords],  # x坐标
        [-coord[1] for coord in sensor_coords],  # y坐标（取负值翻转）
        c=pressure_display_flat,                  # 压力值（颜色）
        cmap='coolwarm',
        vmin=MIN_VALUE,
        vmax=MAX_VALUE,
        marker='s',
        s=200,
        edgecolors='white',
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax1, label='Pressure Intensity')

    # 添加压力值文本
    texts = []
    for i, sensor_coord in enumerate(sensor_coords):
        x, y = sensor_coord
        val = pressure_display_flat[i][0]
        text = ax1.text(x, -y, f"{val:.2f}", ha="center", va="center",
                       fontsize=6, fontweight='bold',
                       color='white' if abs(val) > MAX_VALUE * 0.5 else 'black')
        texts.append(text)

    ax1.set_title(f'Pressure Scatter (Frame {frame_idx + NUM_FRAMES - 1})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # ----- 右侧：角度对比显示 -----
    angle_labels = [
        "Thumb_P", "Thumb_M",
        "Index_P", "Index_M", "Index_I",
        "Middle_P", "Middle_M", "Middle_I",
        "Ring_P", "Ring_M", "Ring_I",
        "Pinky_P", "Pinky_M", "Pinky_I"
    ]

    y_positions = np.linspace(0.9, 0.1, 14)

    ax2.text(0.5, 0.97, "Angle Comparison", fontsize=18, fontweight='bold',
            horizontalalignment='center', transform=ax2.transAxes)

    # 列标题
    ax2.text(0.35, 0.93, "Predicted", fontsize=14, color='red',
            horizontalalignment='center', transform=ax2.transAxes)
    ax2.text(0.65, 0.93, "True", fontsize=14, color='green',
            horizontalalignment='center', transform=ax2.transAxes)
    ax2.text(0.85, 0.93, "Error", fontsize=14, color='blue',
            horizontalalignment='center', transform=ax2.transAxes)

    for i, (label, y_pos) in enumerate(zip(angle_labels, y_positions)):
        # 标签
        ax2.text(0.08, y_pos, f"{label}:", fontsize=12,
                verticalalignment='center', transform=ax2.transAxes)

        # 预测值（红色）
        ax2.text(0.35, y_pos, f"{predicted_angles_deg[i]:.2f}°",
                fontsize=14, fontweight='bold',
                verticalalignment='center', transform=ax2.transAxes,
                color='red')

        # 真实值（绿色）
        ax2.text(0.65, y_pos, f"{true_angles_deg[i]:.2f}°",
                fontsize=14,
                verticalalignment='center', transform=ax2.transAxes,
                color='green')

        # 误差（蓝色）
        ax2.text(0.85, y_pos, f"{errors[i]:.2f}°",
                fontsize=12,
                verticalalignment='center', transform=ax2.transAxes,
                color='blue')

    # 整体MAE
    overall_mae = np.mean(errors)
    ax2.text(0.5, 0.03, f"Overall MAE: {overall_mae:.2f}°", fontsize=16,
            horizontalalignment='center', fontweight='bold',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('single_frame_prediction_scatter.png', dpi=150)
    plt.show()

    # 打印结果
    print("\n" + "="*50)
    print(f"Frame Index: {frame_idx} (using frames {frame_idx} to {frame_idx + NUM_FRAMES - 1})")
    print("="*50)
    print("\nAngle Comparison (degrees):")
    for i, label in enumerate(angle_labels):
        print(f"  {label:12s}: Pred={predicted_angles_deg[i]:.2f}, True={true_angles_deg[i]:.2f}, Error={errors[i]:.2f}")
    print(f"\nOverall MAE: {overall_mae:.2f}°")


if __name__ == '__main__':
    # 可以修改frame_idx来查看不同帧
    predict_and_visualize(frame_idx=300)