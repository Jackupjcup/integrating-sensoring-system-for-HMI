from train_cnn_regression import PressureAngleDataset
from smallCNN_regression import SmallCNN_regression
from IMUProcessing.MPU6050_utils import load_config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
import warnings
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
project_root = Path(r"D:\blender_workplace\modules")
def evaluate_model(model, test_loader, criterion, device, save_dir, pressure_mean, pressure_std):
    """
    在测试集上评估模型
    """

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
        # RMSE
        rmse = math.sqrt(mean_squared_error(true_i, pred_i))
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
    report_txt_path = os.path.join(save_dir, 'pure_evaluation_report.txt')
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
    report_json_path = os.path.join(save_dir, 'pure_evaluation_report.json')
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

    return test_loss, mae_per_angle, overall_mae


def main(dataset_path, random_seed, save_dir):
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    print('Loading dataset...')
    dataset = PressureAngleDataset(dataset_path)

    # 加载测试集
    batch_size = 32
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    # 加载模型
    model = SmallCNN_regression(num_classes=14).to(device)
    model_path = 'train/best_cnn_regression_model.pth'
    # model_path = "D:/blender_workplace/modules/hand_pose_mediapipe/train/Archive/Small_CNN_Multi_Regressor_5Frames/best_cnn_regression_model.pth"

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from epoch {checkpoint["epoch"] + 1}')
        print(f'Model validation loss: {checkpoint["val_loss"]:.6f}')
    else:
        print(f'Warning: Model file {model_path} not found!')
        print('Using untrained model...')

    # 损失函数
    criterion = nn.MSELoss()

    # pressure mean and std
    train_json = load_config("train/evaluation_report.json")
    pressure_mean = train_json["pressure_mean"]
    pressure_std = train_json["pressure_std"]

    # dataset.mean = pressure_mean
    # dataset.std = pressure_std

    # 在测试集上评估
    print('Evaluating on test_on_dataset set...')
    test_loss, mae_per_angle, overall_mae = evaluate_model(model, test_loader, criterion, device, save_dir, pressure_mean, pressure_std)

    # 保存训练结果
    results = {
        'test_loss': test_loss,
        'mae_per_angle': mae_per_angle,
        'overall_mae': overall_mae
    }


if __name__ == "__main__":
    main(dataset_path="record/session_20260413_220616/dataset.csv",
         random_seed=42,
         save_dir="train/test_on_dataset")













