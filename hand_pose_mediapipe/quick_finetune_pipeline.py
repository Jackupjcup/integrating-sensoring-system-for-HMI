"""
快速微调管道：采集数据 -> 制作数据集 -> 训练模型 -> 实时测试

工作流程：
1. 数据采集: 调用 AllData_collect_2.py 的 FingerAngleAndPressureSensorCollector
2. 制作数据集: 参考 assamble_dataset.py，从session目录提取并处理数据
3. 训练模型: 调用 train_cnn_regression.py 的函数
4. 实时测试: 参考 realtime_angle_predict_FlexFrames.py

使用方法:
    python quick_finetune_pipeline.py --epochs 50 --lr 0.001 --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime

# 导入现有模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from AllData_collect_2 import FingerAngleAndPressureSensorCollector
from assamble_dataset import (
    load_df_as_np,
    load_angles_df_as_np,
    normalize_angle_data,
    normalize_pressure_data,
    align_with,
    save_mean_and_std
)
from train_cnn_regression import PressureAngleDataset, train_model, evaluate_model
from smallCNN_regression import SmallCNN_regression


# ============================================================================
# Step 1: 数据采集
# ============================================================================
def collect_data(record_dir="record"):
    """
    调用 FingerAngleAndPressureSensorCollector 采集数据。
    按 'r' 开始/停止录制，按 'q' 退出。
    返回采集的session目录名。
    """
    print("\n" + "="*60)
    print("Step 1: 数据采集")
    print("="*60)
    print("按 'r' 开始/停止录制")
    print("按 'q' 退出")
    print("请采集手势数据（建议采集多种手势，至少10秒）")
    print("="*60 + "\n")

    collector = FingerAngleAndPressureSensorCollector(
        record_dir=record_dir,
        camera_id=0,
        use_threading=True,
        num_hands=2,
        model_asset_path='hand_landmarker.task',
        video_codec='mp4v',
        video_fps=30.0,
        whether_collecting_pressure_data=True
    )

    collector.run()

    # 返回最新的session目录
    sessions = sorted([d for d in os.listdir(record_dir) if d.startswith("session_")])
    if sessions:
        return sessions[-1]
    return None


# ============================================================================
# Step 2: 制作数据集
# ============================================================================
def make_dataset_from_session(record_dir, session_name, ratio_of_part):
    """
    从单个session目录提取数据并制作数据集。
    按ratio_of_part划分：前部分用于训练，后部分用于测试。
    返回 (完整数据集, 训练数据集, 测试数据集, mean_pressure, std_pressure)。
    """
    print("\n" + "="*60)
    print(f"Step 2: 制作数据集 (session: {session_name})")
    print("="*60)

    session_path = os.path.join(record_dir, session_name)
    angles_path = os.path.join(session_path, "angles_data.csv")
    pressure_path = os.path.join(session_path, "pressure_data.csv")

    if not os.path.exists(angles_path):
        print(f"错误: 角度数据文件不存在 {angles_path}")
        return None, None, None

    if not os.path.exists(pressure_path):
        print(f"错误: 压力数据文件不存在 {pressure_path}")
        return None, None, None

    # 加载数据
    angles_data_np = load_angles_df_as_np(angles_path)
    pressure_data_np = load_df_as_np(pressure_path)

    if angles_data_np is None or pressure_data_np is None:
        print("错误: 数据加载失败")
        return None, None, None

    print(f"角度数据: {angles_data_np.shape}")
    print(f"压力数据: {pressure_data_np.shape}")

    # 归一化
    angles_data_np_normalized = normalize_angle_data(angles_data_np.copy(), angle_idx=4)
    pressure_data_np_normalized, mean_pressure, std_pressure = normalize_pressure_data(
        pressure_data_np.copy(), pressure_idx=1
    )

    # 对齐数据
    dataset = align_with(
        camera_data=angles_data_np_normalized,
        sensor_data=pressure_data_np_normalized
    )

    dataset_df = pd.DataFrame(dataset)
    print(f"对齐后数据集: {dataset_df.shape}")

    # 按比例划分：前 ratio_of_part 用于训练，后 1-ratio_of_part 用于测试
    split_idx = int(ratio_of_part * len(dataset_df)) - 1
    train_dataset_df = dataset_df.iloc[:split_idx].copy()
    test_dataset_df = dataset_df.iloc[split_idx:].copy()
    print(f"训练数据集: {train_dataset_df.shape}, 测试数据集: {test_dataset_df.shape}")

    # 保存到test_datasets子目录（避免与原文件混淆）
    test_datasets_dir = os.path.join(session_path, "test_datasets")
    os.makedirs(test_datasets_dir, exist_ok=True)

    dataset_save_path = os.path.join(test_datasets_dir, "dataset.csv")
    dataset_df.to_csv(dataset_save_path, index=False)
    train_dataset_save_path = os.path.join(test_datasets_dir, "train_dataset.csv")
    train_dataset_df.to_csv(train_dataset_save_path, index=False)
    test_dataset_save_path = os.path.join(test_datasets_dir, "test_dataset.csv")
    test_dataset_df.to_csv(test_dataset_save_path, index=False)
    print(f"数据集已保存: {dataset_save_path}")
    print(f"训练数据集已保存: {train_dataset_save_path}")
    print(f"测试数据集已保存: {test_dataset_save_path}")

    # 保存归一化参数
    mean_std_path = os.path.join(test_datasets_dir, "pressure_mean_std.txt")
    save_mean_and_std(mean_pressure, std_pressure, mean_std_path)

    return dataset_df, train_dataset_df, test_dataset_df, mean_pressure, std_pressure


# ============================================================================
# Step 3: 训练模型
# ============================================================================
def finetune_model(dataset_df, mean_pressure, std_pressure, save_dir="train",
                   num_epochs=50, learning_rate=0.001, batch_size=32,
                   train_ratio=0.7, val_ratio=0.15, random_seed=42,
                   pretrained_model_path=None):
    """
    微调训练模型，所有结果保存到 train/quick_finetune 目录。

    参数:
        pretrained_model_path: 预训练模型路径，如果为None则从头训练
    """
    # 所有训练结果保存到 quick_finetune 子目录
    quick_finetune_dir = os.path.join(save_dir, "quick_finetune")
    os.makedirs(quick_finetune_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Step 3: 训练模型")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Train/Val/Test: {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")
    print(f"Pretrained model: {pretrained_model_path if pretrained_model_path else 'None (从头训练)'}")
    print(f"Save directory: {quick_finetune_dir}")
    print("="*60 + "\n")

    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 保存临时数据集文件
    temp_csv_path = os.path.join(save_dir, "temp_finetune_dataset.csv")
    os.makedirs(save_dir, exist_ok=True)
    dataset_df.to_csv(temp_csv_path, index=False)

    # 加载数据集
    dataset = PressureAngleDataset(
        temp_csv_path,
        # mean=mean_pressure,
        # std=std_pressure,
        diff_mean=0.0,
        diff_std=1.0,
        seq_len=5
    )

    # 划分数据集
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 计算训练集的均值和标准差
    train_indices = train_dataset.indices
    train_pressures = dataset.pressures[train_indices]
    pressure_mean = np.mean(train_pressures)
    pressure_std = np.std(train_pressures)
    print(f"Pressure mean: {pressure_mean:.4f}, std: {pressure_std:.4f}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = SmallCNN_regression(num_classes=14).to(device)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model: {pretrained_model_path} (epoch {checkpoint['epoch']+1})")
    else:
        print("No pretrained model, training from scratch.")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        quick_finetune_dir, num_epochs, scheduler
    )

    # 加载最佳模型
    best_model_path = os.path.join(quick_finetune_dir, "best_cnn_regression_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # 评估
    print("\n评估测试集...")
    test_loss, mae_per_angle, overall_mae = evaluate_model(
        model, test_loader, criterion, device, quick_finetune_dir, pressure_mean, pressure_std
    )

    # 训练阶段生成的图片重命名，加 train_ 前缀以区分
    import shutil
    for old_name, new_name in [
        ("residual_plot.png", "train_residual_plot.png"),
        ("prediction_vs_truth_all_angles.png", "train_pred_truth.png"),
    ]:
        src = os.path.join(quick_finetune_dir, old_name)
        dst = os.path.join(quick_finetune_dir, new_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # 保存微调后的模型
    finetuned_model_path = os.path.join(quick_finetune_dir, "fine_tune_model.pth")
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"\n微调模型已保存: {finetuned_model_path}")

    # 删除临时文件
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    return model, train_losses, val_losses


# ============================================================================
# Step 4: 实时测试
# ============================================================================
def run_realtime_test(model_path="train/fine_tune_model.pth"):
    """
    实时测试模型。
    """
    print("\n" + "="*60)
    print("Step 4: 实时测试")
    print("="*60)

    import json
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec
    import threading
    from collections import deque
    from ThreadingCollect_class import ThreadingCollect
    from IMUProcessing.MPU6050_utils import load_config, IMUData
    from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
    from assamble_dataset import raw_data_to_true_img

    # 常量
    ROWS, COLS = 14, 10
    TOTAL_NUM = ROWS * COLS
    UPDATE_INTERVAL = 100
    MAX_VALUE = 500
    MIN_VALUE = -100
    NUM_FRAMES = 5

    # 加载配置
    imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
    palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

    # 配置串口
    import serial
    ser = serial.Serial(
        port=imu_config['serial']['port'],
        baudrate=imu_config['serial']['baud_rate'],
        timeout=imu_config['serial']['timeout']
    )

    # 初始化
    imu_config_path = "D:/blender_workplace/modules/IMUProcessing/config.json"
    imu_1 = IMUData('mpu6050_1', imu_config_path)
    imu_2 = IMUData('mpu6050_2', imu_config_path)
    imu_3 = IMUData('mpu6050_3', imu_config_path)
    palm_points = PalmPointsClass(ROWS, COLS, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

    # 初始化数据收集线程
    threading_collect = ThreadingCollect(
        ser=ser, imu_1=imu_1, imu_2=imu_2, imu_3=imu_3,
        palm_points=palm_points, duration=500
    )

    # 加载归一化参数
    eval_report_path = "train/evaluation_report.json"
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r", encoding='utf-8') as f:
            eval_results = json.load(f)
        pressure_mean = eval_results.get("pressure_mean", 0.0)
        pressure_std = eval_results.get("pressure_std", 1.0)
    else:
        pressure_mean, pressure_std = 0.0, 1.0

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = SmallCNN_regression(num_classes=14).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Warning: Model not found {model_path}")
        return
    model.eval()

    # 帧缓冲区
    frame_buffer = deque(maxlen=NUM_FRAMES)
    frame_count = 0
    for _ in range(NUM_FRAMES):
        frame_buffer.append(np.zeros((ROWS, COLS), dtype=np.float32))

    data_lock = threading.Lock()

    # 创建画布
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # 压力热图
    initial_data = np.zeros(TOTAL_NUM)
    scatter = ax1.scatter(
        [coord[0] for coord in palm_points.plane_pressure_points_coords],
        [-coord[1] for coord in palm_points.plane_pressure_points_coords],
        c=initial_data, cmap='coolwarm',
        vmin=MIN_VALUE, vmax=MAX_VALUE,
        marker='s', s=200, edgecolors='white', linewidth=0.5
    )
    plt.colorbar(scatter, ax=ax1, label='Pressure Intensity')

    # 压力值文本
    texts = []
    for sensor_coord in palm_points.plane_pressure_points_coords:
        x, y = sensor_coord
        text = ax1.text(x, -y, "", ha="center", va="center", fontsize=6, fontweight='bold')
        texts.append(text)

    # 角度显示
    angle_labels = ["P", "M", "index_P", "index_M", "index_I",
                    "middle_P", "middle_M", "middle_I",
                    "ring_P", "ring_M", "ring_I",
                    "pinky_P", "pinky_M", "pinky_I"]
    angle_texts = []
    y_positions = np.linspace(0.9, 0.1, 14)
    for i, (label, y_pos) in enumerate(zip(angle_labels, y_positions)):
        ax2.text(0.08, y_pos, f"{label}:", fontsize=9, verticalalignment='center', transform=ax2.transAxes)
        value_text = ax2.text(0.65, y_pos, "0.00°", fontsize=12, fontweight='bold',
                             verticalalignment='center', transform=ax2.transAxes, color='red')
        angle_texts.append(value_text)

    ax2.text(0.5, 0.95, "Predicted Hand Joint Angles", fontsize=14, fontweight='bold',
            horizontalalignment='center', transform=ax2.transAxes)

    def update(frame):
        nonlocal frame_count

        with data_lock:
            if hasattr(threading_collect, 'palm_points') and threading_collect.palm_points.latest_pressure_data is not None:
                raw_data = threading_collect.palm_points.latest_pressure_data
            else:
                raw_data = np.zeros(TOTAL_NUM)

        scatter.set_array(raw_data)

        for idx in range(TOTAL_NUM):
            val = raw_data[idx]
            texts[idx].set_text(f"{int(val)}")
            texts[idx].set_color("white" if val > MAX_VALUE * 0.6 else "black")

        predicted_angles_np = np.zeros(14)
        if np.sum(np.abs(raw_data)) > 0:
            raw_data_reshaped = raw_data_to_true_img(raw_data.reshape(ROWS, COLS))
            pressure_data = raw_data_reshaped.astype(np.float32)
            pressure_data = (pressure_data - pressure_mean) / (pressure_std + 1e-8)

            frame_buffer.append(pressure_data)
            frame_count += 1

            if frame_count >= NUM_FRAMES:
                buffer_np = np.array(frame_buffer)
                input_tensor = torch.from_numpy(buffer_np).to(device).unsqueeze(0)

                with torch.no_grad():
                    predicted_angles = model(input_tensor)
                predicted_angles_np = predicted_angles.cpu().numpy().flatten()
                predicted_angles_np = np.clip(predicted_angles_np, 0, 1) * 180

        for i, value_text in enumerate(angle_texts):
            angle_value = predicted_angles_np[i]
            value_text.set_text(f"{angle_value:.2f}°")
            if angle_value > 170:
                value_text.set_color('green')
            elif angle_value > 100:
                value_text.set_color('orange')
            else:
                value_text.set_color('red')

        return scatter, *texts, *angle_texts

    # 启动数据收集
    threading_collect.is_collecting = True
    threading_collect.main_thread_flag = True
    threading_collect.stop_event.clear()
    threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
    threading_collect.t1.start()

    print("启动实时可视化...")
    ani = FuncAnimation(fig=fig, func=update, frames=np.arange(0, 10000),
                       interval=UPDATE_INTERVAL, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

    print("实时测试结束。")


# ============================================================================
# Main
# ============================================================================
def main(
    # 数据采集参数
    record_dir="record",
    skip_collection=False,

    # 数据集参数
    session_name=None,

    # 训练参数
    num_epochs=50,
    learning_rate=0.001,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    random_seed=42,
    save_dir="train",
    pretrained_model_path="train/best_cnn_regression_model.pth",

    # 测试参数
    run_test=True,
    run_test_on_dataset=False,
    ratio_of_part=0.3
):
    """
    完整的微调管道。
    """
    print("\n" + "="*60)
    print("快速微调管道")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: 数据采集
    if not skip_collection:
        new_session = collect_data(record_dir)
        if new_session:
            session_name = new_session
        else:
            print("错误: 采集失败")
            return
    else:
        if session_name is None:
            session_name = "session_20260327_170436"
            # print("错误: 没有找到session")


    print(f"使用session: {session_name}")

    # Step 2: 制作数据集
    if not run_test_on_dataset:
        ratio_of_part = 1.0
    dataset_df, train_dataset_df, test_dataset_df, mean_pressure, std_pressure = make_dataset_from_session(record_dir, session_name, ratio_of_part)
    if dataset_df is None:
        print("错误: 数据集制作失败")
        return

    # Step 3: 训练模型 (使用前 ratio_of_part 部分的数据)
    model, train_losses, val_losses = finetune_model(
        train_dataset_df, mean_pressure, std_pressure,
        save_dir=save_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
        pretrained_model_path=pretrained_model_path
    )

    # Step 4: 在测试集上评估 (使用后 1-ratio_of_part 部分的数据)
    if run_test_on_dataset:
        print("\n" + "="*60)
        print("Step 4: 在数据集上测试")
        print("="*60)
        print(f"测试数据集大小: {len(test_dataset_df)}")

        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 保存测试数据集到临时CSV，供PressureAngleDataset加载
        temp_test_csv_path = os.path.join(save_dir, "temp_test_dataset.csv")
        test_dataset_df.to_csv(temp_test_csv_path, index=False)

        # 加载测试数据集
        test_dataset = PressureAngleDataset(
            temp_test_csv_path,
            # mean=mean_pressure,
            # std=std_pressure,
            diff_mean=0.0,
            diff_std=1.0,
            seq_len=5
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 加载训练好的模型 (从 quick_finetune 目录)
        quick_finetune_dir = os.path.join(save_dir, "quick_finetune")
        model_path = os.path.join(quick_finetune_dir, "fine_tune_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model: {model_path}")
        else:
            print(f"Warning: Model not found {model_path}, using current trained model")

        # 评估 - 保存到 quick_finetune 目录
        criterion = nn.MSELoss()
        os.makedirs(quick_finetune_dir, exist_ok=True)

        # 保存训练数据集到临时CSV用于统计计算
        temp_train_csv_path = os.path.join(save_dir, "temp_train_dataset.csv")
        train_dataset_df.to_csv(temp_train_csv_path, index=False)

        train_dataset_for_stats = PressureAngleDataset(
            temp_train_csv_path,
            mean=mean_pressure,
            std=std_pressure,
            diff_mean=0.0,
            diff_std=1.0,
            seq_len=5
        )
        train_pressures = train_dataset_for_stats.pressures
        pressure_mean_for_eval = np.mean(train_pressures)
        pressure_std_for_eval = np.std(train_pressures)

        # 调用test_model_on_dataset.py中的evaluate_model函数
        from test_model_on_dataset import evaluate_model as evaluate_on_test_set
        test_loss, mae_per_angle, overall_mae = evaluate_on_test_set(
            model, test_loader, criterion, device, quick_finetune_dir,
            pressure_mean_for_eval, pressure_std_for_eval
        )

        # 将测试阶段生成的图片重命名，加 test_ 前缀以区分
        import shutil
        for old_name, new_name in [
            ("residual_plot.png", "test_residual_plot.png"),
            ("prediction_vs_truth_all_angles.png", "dataset_pred_truth.png"),
            ("pure_evaluation_report.txt", "test_evaluation_report.txt"),
            ("pure_evaluation_report.json", "test_evaluation_report.json"),
        ]:
            src = os.path.join(quick_finetune_dir, old_name)
            dst = os.path.join(quick_finetune_dir, new_name)
            if os.path.exists(src):
                shutil.move(src, dst)

        # 删除临时文件
        if os.path.exists(temp_test_csv_path):
            os.remove(temp_test_csv_path)
        if os.path.exists(temp_train_csv_path):
            os.remove(temp_train_csv_path)

        print("\n" + "="*60)
        print("数据集测试完成!")
        print(f"测试损失: {test_loss:.6f}")
        print(f"整体MAE: {overall_mae:.6f}")
        print(f"评估报告已保存至: {quick_finetune_dir}")
        print("="*60)

    # Step 5: 实时测试 (可选)
    if run_test:
        user_input = input("\n是否进行实时测试? (Y/n): ").strip().lower()
        if user_input != 'n':
            quick_finetune_dir = os.path.join(save_dir, "quick_finetune")
            run_realtime_test(model_path=os.path.join(quick_finetune_dir, "fine_tune_model.pth"))

    print("\n" + "="*60)
    print("微调管道完成!")
    print(f"模型已保存: {os.path.join(save_dir, 'fine_tune_model.pth')}")
    print("="*60)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='快速微调管道')

    # 数据采集参数
    parser.add_argument('--record-dir', type=str, default='record',
                        help='数据存储目录')
    parser.add_argument('--skip-collection', action='store_true',
                        help='跳过数据采集，使用已有session')
    parser.add_argument('--session', type=str, default=None,
                        help='指定的session名称 (跳过采集时使用)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save-dir', type=str, default='train',
                        help='模型保存目录')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练模型路径(.pth)，不指定则从头训练')

    # 测试参数
    parser.add_argument('--no-test', action='store_true',
                        help='训练后不进行实时测试')

    args = parser.parse_args()

    main(
        record_dir=args.record_dir,
        skip_collection=True,
        session_name=args.session,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        save_dir=args.save_dir,
        pretrained_model_path=args.pretrained,
        run_test=False,
        run_test_on_dataset=True,
        ratio_of_part=0.5
    )