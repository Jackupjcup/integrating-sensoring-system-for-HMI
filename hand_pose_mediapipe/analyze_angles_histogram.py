"""
分析total_dataset.csv中每个angles所在区间的直方图
提取angles列（14个角度值），为每个角度创建直方图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import json
import math
from pathlib import Path

warnings.filterwarnings('ignore')

def parse_angles_string(angles_str):
    """
    解析angles字符串为numpy数组
    格式类似："[0.965431 0.947796 0.935992 ...]"
    """
    # 移除所有括号和换行符，用空格替换
    angles_str = angles_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    # 移除多余空格，分割字符串
    angles_array = np.fromstring(angles_str, sep=' ', dtype=np.float32)
    return angles_array

def load_angles_data(csv_path):
    """
    从CSV文件加载所有angles数据
    """
    print(f"Loading data from {csv_path}...")

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # 解析angles列
    all_angles = []
    angles_count = 0

    for idx, row in df.iterrows():
        angles_str = row['angles']
        try:
            angles_array = parse_angles_string(angles_str)
            if len(angles_array) != 14:
                print(f"Warning: Row {idx} has {len(angles_array)} angles, expected 14")
                continue
            all_angles.append(angles_array)
            angles_count += 1
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            continue

    # 转换为numpy数组 (n_samples, 14)
    all_angles = np.array(all_angles)
    print(f"Successfully parsed {angles_count} samples, shape: {all_angles.shape}")

    return all_angles

def analyze_angles_distribution(angles_data, output_dir='angles_histograms'):
    """
    分析每个角度的分布并创建直方图
    angles_data: (n_samples, 14) numpy数组
    """
    os.makedirs(output_dir, exist_ok=True)

    num_angles = angles_data.shape[1]
    n_samples = angles_data.shape[0]

    print(f"\nAnalyzing distribution of {num_angles} angles from {n_samples} samples...")

    # 计算基本统计信息
    stats_per_angle = []
    for i in range(num_angles):
        angle_data = angles_data[:, i]
        stats = {
            'angle_idx': i,
            'mean': float(np.mean(angle_data)),
            'std': float(np.std(angle_data)),
            'min': float(np.min(angle_data)),
            'max': float(np.max(angle_data)),
            'median': float(np.median(angle_data)),
            'q25': float(np.percentile(angle_data, 25)),
            'q75': float(np.percentile(angle_data, 75)),
            'range': float(np.max(angle_data) - np.min(angle_data))
        }
        stats_per_angle.append(stats)

    # 保存统计信息为JSON
    stats_path = os.path.join(output_dir, 'angles_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_per_angle, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to {stats_path}")

    # 创建文本报告
    report_path = os.path.join(output_dir, 'angles_distribution_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('Angles Distribution Analysis Report\n')
        f.write('='*70 + '\n\n')
        f.write(f'Total samples: {n_samples}\n')
        f.write(f'Number of angles per sample: {num_angles}\n\n')

        f.write('Statistics per angle:\n')
        f.write('-'*70 + '\n')
        for stats in stats_per_angle:
            i = stats['angle_idx']
            f.write(f'Angle {i+1}:\n')
            f.write(f'  Mean:     {stats["mean"]:.6f}\n')
            f.write(f'  Std:      {stats["std"]:.6f}\n')
            f.write(f'  Min:      {stats["min"]:.6f}\n')
            f.write(f'  Max:      {stats["max"]:.6f}\n')
            f.write(f'  Median:   {stats["median"]:.6f}\n')
            f.write(f'  Q25:      {stats["q25"]:.6f}\n')
            f.write(f'  Q75:      {stats["q75"]:.6f}\n')
            f.write(f'  Range:    {stats["range"]:.6f}\n')
            f.write('-'*70 + '\n')

    print(f"Report saved to {report_path}")

    # ==================== 可视化 ====================
    # 1. 单个角度的直方图（14个独立图）
    print("\nCreating individual histograms...")
    for i in range(num_angles):
        angle_data = angles_data[:, i]

        plt.figure(figsize=(10, 6))

        # 计算合适的bins数量（使用Freedman-Diaconis规则）
        iqr = stats_per_angle[i]['q75'] - stats_per_angle[i]['q25']
        bin_width = 2 * iqr / (n_samples ** (1/3))
        if bin_width == 0:
            bins = 30
        else:
            bins = int((stats_per_angle[i]['max'] - stats_per_angle[i]['min']) / bin_width)
            bins = max(10, min(bins, 50))

        # 绘制直方图
        n, bins, patches = plt.hist(angle_data, bins=bins, alpha=0.7,
                                   color='skyblue', edgecolor='black', linewidth=0.5)

        # 添加统计信息文本
        stats_text = f'Mean: {stats_per_angle[i]["mean"]:.4f}\n'
        stats_text += f'Std: {stats_per_angle[i]["std"]:.4f}\n'
        stats_text += f'Min: {stats_per_angle[i]["min"]:.4f}\n'
        stats_text += f'Max: {stats_per_angle[i]["max"]:.4f}\n'
        stats_text += f'Median: {stats_per_angle[i]["median"]:.4f}\n'
        stats_text += f'Samples: {n_samples}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.xlabel('Angle Degree')
        plt.ylabel('Frequency')
        plt.title(f'Angle {i+1} Distribution (Histogram)')
        plt.grid(True, alpha=0.3)

        # 保存图像
        hist_path = os.path.join(output_dir, f'angle_{i+1}_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Individual histograms saved to {output_dir}/angle_*_histogram.png")

    # 2. 所有角度的直方图组合（2×7布局）
    print("\nCreating combined histogram plot (2×7 layout)...")
    fig, axes = plt.subplots(2, 7, figsize=(21, 10))
    axes = axes.flatten()

    for i in range(num_angles):
        ax = axes[i]
        angle_data = angles_data[:, i]

        # 计算bins
        iqr = stats_per_angle[i]['q75'] - stats_per_angle[i]['q25']
        bin_width = 2 * iqr / (n_samples ** (1/3))
        if bin_width == 0:
            bins = 30
        else:
            bins = int((stats_per_angle[i]['max'] - stats_per_angle[i]['min']) / bin_width)
            bins = max(10, min(bins, 30))

        # 绘制直方图
        ax.hist(angle_data, bins=bins, alpha=0.7, color='lightcoral',
                edgecolor='black', linewidth=0.5)

        # 添加均值和标准差线
        mean_val = stats_per_angle[i]['mean']
        std_val = stats_per_angle[i]['std']
        ax.axvline(mean_val, color='blue', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(mean_val + std_val, color='green', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Angle {i+1}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # 隐藏多余的子图（如果有）
    for i in range(num_angles, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Angles Distribution Histograms (All 14 Angles)', fontsize=16, y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'all_angles_histograms_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined histogram plot saved to {combined_path}")

    # 3. 箱线图（Box Plot）比较所有角度
    print("\nCreating box plot for comparison...")
    plt.figure(figsize=(14, 8))
    box_data = [angles_data[:, i] for i in range(num_angles)]
    box_labels = [f'Angle {i+1}' for i in range(num_angles)]

    box = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, num_angles))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Angle Index')
    plt.ylabel('Degree')
    plt.title('Angles Distribution Comparison (Box Plot)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

    boxplot_path = os.path.join(output_dir, 'angles_boxplot_comparison.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved to {boxplot_path}")

    # 4. 密度图（KDE）组合
    print("\nCreating density plot (KDE)...")
    fig, axes = plt.subplots(2, 7, figsize=(21, 10))
    axes = axes.flatten()

    for i in range(num_angles):
        ax = axes[i]
        angle_data = angles_data[:, i]

        # 绘制KDE
        from scipy import stats
        try:
            kde = stats.gaussian_kde(angle_data)
            x_range = np.linspace(stats_per_angle[i]['min'], stats_per_angle[i]['max'], 200)
            y_kde = kde(x_range)
            ax.plot(x_range, y_kde, 'b-', linewidth=2, label='KDE')
        except:
            # 如果KDE失败，使用直方图归一化
            ax.hist(angle_data, bins=30, density=True, alpha=0.5, color='lightblue')

        # 添加均值和标准差线
        mean_val = stats_per_angle[i]['mean']
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.3f}')

        ax.set_xlabel('Degree')
        ax.set_ylabel('Density')
        ax.set_title(f'Angle {i+1} (KDE)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # 隐藏多余的子图
    for i in range(num_angles, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Angles Density Estimation (KDE)', fontsize=16, y=1.02)
    plt.tight_layout()
    kde_path = os.path.join(output_dir, 'angles_kde_density.png')
    plt.savefig(kde_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Density plot saved to {kde_path}")

    print(f"\nAll visualizations saved to directory: {output_dir}")
    return stats_per_angle

def main():
    """主函数"""
    csv_path = r'D:\blender_workplace\modules\hand_pose_mediapipe\record\total_dataset.csv'
    output_dir = 'angles_histograms'

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    try:
        # 加载角度数据
        normalized_angles_data = load_angles_data(csv_path)

        angles_data = normalized_angles_data * 180


        # 分析分布并创建可视化
        stats = analyze_angles_distribution(angles_data, output_dir)

        print("\n" + "="*70)
        print("Analysis completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*70)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()