"""
训练CNN回归模型，从压力传感器数据（14x10）预测手部关节角度（14个角度）
使用SmallCNN_regression模型架构。
数据集：dataset.csv，包含pressure和angles列。
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import json
import math
from pathlib import Path

warnings.filterwarnings('ignore')

project_root = Path(r"D:\blender_workplace\modules")
sys.path.append(str(project_root))

# 导入自定义模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smallCNN_regression import SmallCNN_regression

# 自定义数据集类
class PressureAngleDataset(Dataset):
    def __init__(self, csv_path, mean=None, std=None, diff_mean=None, diff_std=None, seq_len=5):
        """
        读取dataset.csv，解析pressure和angles列
        pressure: 字符串表示的14x10 numpy数组
        angles: 字符串表示的14维numpy数组
        mean, std: 可选的标准化参数，如果提供则应用标准化
        """
        self.data = pd.read_csv(csv_path)
        self.mean = mean
        self.std = std
        self.diff_mean = diff_mean
        self.diff_std = diff_std
        self.seq_len = seq_len

        # 解析pressure和angles字符串为numpy数组
        self.pressures = []
        self.angles = []


        print("Parsing dataset...")
        for idx in range(len(self.data)):
            # 解析pressure
            pressure_str = self.data.iloc[idx]['pressure']
            # 移除所有括号和换行符，用空格替换
            pressure_str = pressure_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
            # 移除多余空格，分割字符串
            pressure_array = np.fromstring(pressure_str, sep=' ', dtype=np.float32)
            pressure_array = pressure_array.reshape(14, 10)

            # 解析angles
            angles_str = self.data.iloc[idx]['angles']
            angles_str = angles_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
            angles_array = np.fromstring(angles_str, sep=' ', dtype=np.float32)

            self.pressures.append(pressure_array)
            self.angles.append(angles_array)

        # 转换为numpy数组
        self.pressures = np.array(self.pressures)
        self.angles = np.array(self.angles)

        all_diff = []

        for i in range(len(self.pressures) - 1):
            d = self.pressures[i + 1] - self.pressures[i]
            all_diff.append(d)

        all_diff = np.stack(all_diff, axis=0)

        # self.diff_mean = all_diff.mean()
        # self.diff_std = all_diff.std()

        print(f"Dataset loaded: {self.pressures.shape} pressures, {self.angles.shape} angles")
        print(f"mean {self.diff_mean}, std {self.diff_std}")

    # def __len__(self):
    #     return len(self.pressures)
    #
    # def __getitem__(self, idx):
    #     pressure = self.pressures[idx]
    #     angle = self.angles[idx]
    #
    #     # 应用标准化（如果提供了mean和std）
    #     if self.mean is not None and self.std is not None:
    #         pressure = (pressure - self.mean) / (self.std + 1e-8)
    #
    #     # 转换为torch张量
    #     pressure_tensor = torch.from_numpy(pressure).unsqueeze(0)  # 增加通道维度 (1, 14, 10)
    #     angle_tensor = torch.from_numpy(angle)
    #
    #     return pressure_tensor, angle_tensor

    ''' 5 Consecutive Frames '''
    def __len__(self):
        return len(self.pressures) - self.seq_len + 1

    def __getitem__(self, idx):

        # 连续5帧
        x = self.pressures[idx:idx + self.seq_len]  # (5,14,10)

        x_diff = np.diff(x, axis=0)

        # 应用标准化（如果提供了mean和std）
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        if self.diff_mean is not None and self.diff_std is not None:
            x_diff = (x_diff - self.diff_mean) / (self.diff_std + 1e-8)

        # concatenate original x and diff x alternatively
        combined = []
        for i in range(self.seq_len - 1):
            combined.append(x[i])  # x_t
            combined.append(x_diff[i])  # Δ_t
        combined.append(x[-1])  # 最后一帧

        # (9,14,10)
        x_combined = np.stack(combined, axis=0)

        x_centered = x - x.mean(axis=(1,2), keepdims=True)

        # x_total = np.concatenate([x_combined, x_centered], axis=0)

        # label = 最后一帧
        y = self.angles[idx + self.seq_len - 1]  # (14,)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # print(f"x_tensor shape is {x_tensor.shape}, y_tensor shape is {y_tensor.shape}")

        return x_tensor, y_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, device, save_dir, num_epochs=50, scheduler=None):
    """
    训练模型
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_cnn_regression_model.pth")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, best_model_path)
            print(f'  Best model saved with val loss: {best_val_loss:.6f}')

    # 绘制损失曲线
    png_save_path = os.path.join(save_dir, 'training_loss_curve.png')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(png_save_path)
    plt.show()

    return train_losses, val_losses


def visualize_sample_prediction(model, dataset, device, sample_idx=0):
    """
    可视化单个样本的压力热图和预测角度
    类似PressureHeat_RowColsLines.py布局，左边压力热图，右边预测角度
    """
    import matplotlib.gridspec as gridspec

    # 获取原始压力数据（处理Subset对象）
    if hasattr(dataset, 'dataset'):  # 如果是Subset对象
        original_dataset = dataset.dataset
        original_idx = dataset.indices[sample_idx]
        raw_pressure_data = original_dataset.pressures[original_idx]  # 原始数据
    elif hasattr(dataset, 'pressures'):  # 如果是原始数据集
        raw_pressure_data = dataset.pressures[sample_idx]
        original_idx = sample_idx
    else:
        raw_pressure_data = None

    # 获取标准化后的数据用于模型预测
    pressure_tensor, true_angles = dataset[sample_idx]
    pressure_data_for_model = pressure_tensor.squeeze(0).numpy()  # (14, 10)

    # 模型预测
    model.eval()
    with torch.no_grad():
        input_tensor = pressure_tensor.unsqueeze(0).to(device)  # (1, 1, 14, 10)
        predicted_angles = model(input_tensor)
        predicted_angles = predicted_angles.cpu().numpy().flatten()

    true_angles = true_angles.numpy()

    # 使用原始压力数据进行可视化（如果可用），否则使用标准化后的数据
    if raw_pressure_data is not None:
        pressure_data_for_viz = raw_pressure_data
    else:
        pressure_data_for_viz = pressure_data_for_model
        print("Note: Using normalized pressure data for visualization")

    # 创建画布
    fig = plt.figure(figsize=(12, 6))

    # 使用gridspec划分布局：1行2列
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    # 创建两个轴对象
    ax1 = fig.add_subplot(gs[0, 0])  # 左侧压力热图
    ax2 = fig.add_subplot(gs[0, 1])  # 右侧角度显示

    # 隐藏右侧轴的坐标轴
    ax2.axis('off')

    # ---------------------- 左侧：压力热图 ----------------------
    # 显示压力热图
    im = ax1.imshow(pressure_data_for_viz, cmap='coolwarm', aspect='auto',
                    vmin=pressure_data_for_viz.min(), vmax=pressure_data_for_viz.max())
    plt.colorbar(im, ax=ax1, label='Pressure Intensity')

    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    ax1.set_title(f'Pressure Heatmap (Sample {sample_idx})')

    # 添加网格线
    ax1.set_xticks(np.arange(pressure_data_for_viz.shape[1]))
    ax1.set_yticks(np.arange(pressure_data_for_viz.shape[0]))
    ax1.grid(which='both', color='white', linestyle='-', linewidth=0.5)

    # 显示压力值（可选，如果数据不多可以显示）
    if pressure_data_for_viz.shape[0] <= 20 and pressure_data_for_viz.shape[1] <= 20:
        for i in range(pressure_data_for_viz.shape[0]):
            for j in range(pressure_data_for_viz.shape[1]):
                ax1.text(j, i, f'{pressure_data_for_viz[i, j]:.1f}',
                        ha='center', va='center', fontsize=6,
                        color='white' if abs(pressure_data_for_viz[i, j]) > pressure_data_for_viz.max() * 0.5 else 'black')

    # ---------------------- 右侧：角度显示 ----------------------
    # 角度标签
    angle_labels = [f'Angle {i+1}' for i in range(14)]

    # y位置均匀分布
    y_positions = np.linspace(0.9, 0.1, 14)

    # 显示预测角度和真实角度
    for i, (label, y_pos) in enumerate(zip(angle_labels, y_positions)):
        # 标签
        ax2.text(0.05, y_pos, f"{label}:", fontsize=10,
                verticalalignment='center', transform=ax2.transAxes)

        # 预测值（红色）
        pred_text = ax2.text(0.35, y_pos, f"{predicted_angles[i]:.1f}°",
                            fontsize=11, fontweight='bold',
                            verticalalignment='center', transform=ax2.transAxes,
                            color='red')

        # 真实值（绿色，如果有）
        true_text = ax2.text(0.65, y_pos, f"({true_angles[i]:.1f}°)",
                            fontsize=10,
                            verticalalignment='center', transform=ax2.transAxes,
                            color='green')

        # 误差
        error = abs(predicted_angles[i] - true_angles[i])
        error_text = ax2.text(0.85, y_pos, f"[{error:.1f}]",
                             fontsize=9,
                             verticalalignment='center', transform=ax2.transAxes,
                             color='blue')

    # 添加标题和说明
    ax2.text(0.5, 0.95, "Angle Predictions", fontsize=14, fontweight='bold',
            horizontalalignment='center', transform=ax2.transAxes)
    ax2.text(0.35, 0.97, "(Predicted)", fontsize=9, color='red',
            horizontalalignment='center', transform=ax2.transAxes)
    ax2.text(0.65, 0.97, "(True)", fontsize=9, color='green',
            horizontalalignment='center', transform=ax2.transAxes)
    ax2.text(0.85, 0.97, "[Error]", fontsize=9, color='blue',
            horizontalalignment='center', transform=ax2.transAxes)

    # 计算整体MAE
    overall_mae = np.mean(np.abs(predicted_angles - true_angles))
    ax2.text(0.5, 0.03, f"Overall MAE: {overall_mae:.2f}°", fontsize=11,
            horizontalalignment='center', fontweight='bold',
            transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('train/sample_prediction_visualization.png')
    plt.show()

    print(f"Visualized sample {sample_idx}")
    print(f"Predicted angles: {predicted_angles}")
    print(f"True angles: {true_angles}")
    print(f"Overall MAE: {overall_mae:.2f}°")


def evaluate_model(model, test_loader, criterion, device, save_dir, pressure_mean, pressure_std):
    """
    在测试集上评估模型
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    running_loss = 0.0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            predictions.append(outputs.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    # 计算每个角度的平均绝对误差
    mae_per_angle = np.mean(np.abs(predictions - ground_truths), axis=0)
    overall_mae = np.mean(mae_per_angle)

    print(f'Test Loss: {test_loss:.6f}')
    print(f'Overall MAE: {overall_mae:.6f}')
    print('MAE per angle:')
    for i, mae in enumerate(mae_per_angle):
        print(f'  Angle {i+1}: {mae:.6f}')

    # ==================== 新增：完整评估指标 ====================
    # 计算每个角度的R², RMSE, MAPE
    num_angles = predictions.shape[1]
    r2_per_angle = []
    rmse_per_angle = []
    mape_per_angle = []

    for i in range(num_angles):
        pred_i = predictions[:, i]
        true_i = ground_truths[:, i]
        # R² Score
        r2 = r2_score(true_i, pred_i)
        r2_per_angle.append(r2)
        mse = mean_squared_error(true_i, pred_i)
        # RMSE
        rmse = math.sqrt(mse)
        rmse_per_angle.append(rmse)
        # MAPE (Mean Absolute Percentage Error)
        # 避免除零，仅计算非零真实值
        mask = true_i != 0
        if np.any(mask):
            mape = np.mean(np.abs((true_i[mask] - pred_i[mask]) / true_i[mask])) * 100
        else:
            mape = np.nan
        mape_per_angle.append(mape)

    # 整体指标（跨所有样本和角度）
    predictions_flat = predictions.flatten()
    ground_truths_flat = ground_truths.flatten()
    overall_r2 = r2_score(ground_truths_flat, predictions_flat)
    overall_rmse = math.sqrt(mean_squared_error(ground_truths_flat, predictions_flat))
    overall_mse = mean_squared_error(ground_truths_flat, predictions_flat)
    # 整体MAPE
    mask = ground_truths_flat != 0
    if np.any(mask):
        overall_mape = np.mean(np.abs((ground_truths_flat[mask] - predictions_flat[mask]) / ground_truths_flat[mask])) * 100
    else:
        overall_mape = np.nan

    print('\n' + '='*60)
    print('扩展评估指标:')
    print('='*60)
    print(f'整体 R² Score: {overall_r2:.6f}')
    print(f'整体 MSE: {overall_mse:.6f}')
    print(f'整体 RMSE: {overall_rmse:.6f}')
    print(f'整体 MAPE: {overall_mape:.6f}%' if not np.isnan(overall_mape) else f'整体 MAPE: NaN (真实值全为零)')
    print('\n每个角度的指标:')
    for i in range(num_angles):
        print(f'  Angle {i+1}: R²={r2_per_angle[i]:.6f}, RMSE={rmse_per_angle[i]:.6f}, MAPE={mape_per_angle[i]:.6f}%'
              if not np.isnan(mape_per_angle[i]) else f'  Angle {i+1}: R²={r2_per_angle[i]:.6f}, RMSE={rmse_per_angle[i]:.6f}, MAPE=NaN')

    # ==================== 保存报告文件 ====================
    # 文本报告
    report_txt_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('CNN回归模型评估报告\n')
        f.write('='*60 + '\n\n')
        f.write(f'测试集大小: {len(test_loader.dataset)}\n')
        f.write(f'pressure_mean: {pressure_mean}\n')
        f.write(f'pressure_std: {pressure_std}\n')
        f.write(f'测试损失 (MSE): {test_loss:.6f}\n')
        f.write(f'整体 MAE: {overall_mae:.6f}\n')
        f.write(f'整体 R² Score: {overall_r2:.6f}\n')
        f.write(f'整体 RMSE: {overall_rmse:.6f}\n')
        if not np.isnan(overall_mape):
            f.write(f'整体 MAPE: {overall_mape:.6f}%\n')
        else:
            f.write('整体 MAPE: NaN (真实值全为零)\n')
        f.write('\n每个角度的详细指标:\n')
        for i in range(num_angles):
            f.write(f'  Angle {i+1}:\n')
            f.write(f'    MAE: {mae_per_angle[i]:.6f}\n')
            f.write(f'    R²: {r2_per_angle[i]:.6f}\n')
            f.write(f'    RMSE: {rmse_per_angle[i]:.6f}\n')
            if not np.isnan(mape_per_angle[i]):
                f.write(f'    MAPE: {mape_per_angle[i]:.6f}%\n')
            else:
                f.write('    MAPE: NaN\n')
        f.write('\n' + '='*60 + '\n')
        f.write('报告生成时间: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('='*60 + '\n')

    # JSON报告
    report_json_path = os.path.join(save_dir, 'evaluation_report.json')
    report_data = {
        'pressure_mean': float(pressure_mean),
        'pressure_std': float(pressure_std),
        'test_loss': float(test_loss),
        'overall_mae': float(overall_mae),
        'overall_r2': float(overall_r2),
        'overall_rmse': float(overall_rmse),
        'overall_mape': float(overall_mape) if not np.isnan(overall_mape) else None,
        'per_angle': {
            f'angle_{i+1}': {
                'mae': float(mae_per_angle[i]),
                'r2': float(r2_per_angle[i]),
                'rmse': float(rmse_per_angle[i]),
                'mape': float(mape_per_angle[i]) if not np.isnan(mape_per_angle[i]) else None
            } for i in range(num_angles)
        }
    }
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f'\n评估报告已保存至:\n  {report_txt_path}\n  {report_json_path}')

    # ==================== 可视化 ====================
    # 1. 残差图 (Residual Plot)
    residuals = predictions_flat - ground_truths_flat
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions_flat, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual (Predict - Truth)')
    plt.title('Residual Plot (All Angles)')
    plt.grid(True, alpha=0.3)
    residual_plot_path = os.path.join(save_dir, 'residual_plot.png')
    plt.savefig(residual_plot_path, dpi=150)
    plt.show()

    # 2. 预测 vs 真实值散点图（14个角度，2×7布局）
    fig, axes = plt.subplots(2, 7, figsize=(21, 8))
    axes = axes.flatten()

    for i in range(num_angles):
        ax = axes[i]
        pred_i = predictions[:, i]
        true_i = ground_truths[:, i]
        ax.scatter(true_i, pred_i, alpha=0.5, s=10)
        # 绘制对角线
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Angle {i+1}')
        ax.grid(True, alpha=0.3)
        # 在子图中显示R²和MAE
        ax.text(0.05, 0.95, f'R²={r2_per_angle[i]:.3f}\nMAE={mae_per_angle[i]:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的子图（如果有）
    for i in range(num_angles, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Predict vs Truth (14 angles)', fontsize=16, y=1.02)
    plt.tight_layout()
    pred_vs_truth_path = os.path.join(save_dir, 'prediction_vs_truth_all_angles.png')
    plt.savefig(pred_vs_truth_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'可视化已保存至:\n  {residual_plot_path}\n  {pred_vs_truth_path}')

    # 3. 保留原有的单个角度图（可选，可注释掉）
    # png_save_path = os.path.join(save_dir, 'prediction_vs_truth_angle1.png')
    # plt.figure(figsize=(10, 5))
    # plt.scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.5)
    # plt.plot([ground_truths[:, 0].min(), ground_truths[:, 0].max()],
    #          [ground_truths[:, 0].min(), ground_truths[:, 0].max()], 'r--')
    # plt.xlabel('Ground Truth (Angle 1)')
    # plt.ylabel('Prediction (Angle 1)')
    # plt.title('Prediction vs Ground Truth for Angle 1')
    # plt.grid(True)
    # plt.savefig(png_save_path)
    # plt.show()

    # 新增：可视化单个样本的压力热图和预测角度
    # 选取测试集中的第一个样本进行可视化
    # if len(test_loader.dataset) > 0:
    #     visualize_sample_prediction(model, test_loader.dataset, device)

    return test_loss, mae_per_angle, overall_mae

def main(dataset_path, random_seed, save_dir, num_epoch=100):
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据集路径
    dataset_path = dataset_path
    best_model_path = os.path.join(save_dir, 'best_cnn_regression_model.pth')
    final_model_path = os.path.join(save_dir, 'final_cnn_regression_model.pth')
    training_result_path = os.path.join(save_dir, 'training_results.npy')

    # 加载数据集
    print('Loading dataset...')
    dataset = PressureAngleDataset(dataset_path)

    # 划分数据集：70% 训练，15% 验证，15% 测试
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}')

    # 计算训练集的均值和标准差（用于标准化）
    print('Computing normalization statistics from training set...')
    train_indices = train_dataset.indices
    train_pressures = dataset.pressures[train_indices]  # shape: (n_train, 14, 10)
    pressure_mean = np.mean(train_pressures)
    pressure_std = np.std(train_pressures)
    print(f'Pressure mean: {pressure_mean:.4f}, std: {pressure_std:.4f}')

    # # 设置数据集的标准化参数
    # dataset.mean = pressure_mean
    # dataset.std = pressure_std

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = SmallCNN_regression(num_classes=14).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练模型
    print('Starting training...')
    num_epochs = num_epoch
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, save_dir, num_epochs, scheduler)

    # 加载最佳模型进行测试
    # best_model_path = 'train/best_cnn_regression_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded best model from epoch {checkpoint["epoch"]+1}')

    # 在测试集上评估
    print('Evaluating on test_on_dataset set...')
    test_loss, mae_per_angle, overall_mae = evaluate_model(model, test_loader, criterion, device, save_dir, pressure_mean, pressure_std)

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved as {final_model_path}')

    # 保存训练结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'mae_per_angle': mae_per_angle,
        'overall_mae': overall_mae
    }
    np.save(training_result_path, results)
    print(f'Training results saved to {training_result_path}')



if __name__ == '__main__':
    main(dataset_path=r'D:\blender_workplace\modules\hand_pose_mediapipe\record\total_dataset.csv',
         random_seed=42,
         save_dir="train",
         num_epoch=200)









