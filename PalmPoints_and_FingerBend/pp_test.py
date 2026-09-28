import numpy as np
from IPython.core.pylabtools import figsize
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def denoise_pressure_data(raw_data,
                          time_smooth_window=5,  # 时间维度滑动平均窗口
                          spatial_median_size=3,  # 空间中值滤波核大小
                          gaussian_sigma=1,  # 空间高斯滤波sigma
                          detrend=True):  # 是否去除线性趋势
    """
    对压力传感器时间序列数据进行组合去噪
    参数:
        raw_data: 原始数据，形状为 (num_frame, 14, 10)
        time_smooth_window: 时间维度滑动平均窗口大小（奇数）
        spatial_median_size: 空间中值滤波核大小（奇数）
        gaussian_sigma: 高斯滤波标准差
        detrend: 是否去除线性趋势（消除基线漂移）
    返回:
        denoised_data: 去噪后的数据，形状与输入一致
    """
    # 1. 复制数据避免修改原数据
    data = raw_data.copy().astype(np.float32)
    num_frame, h, w = data.shape

    # 2. 去除线性趋势（解决基线缓慢漂移）
    if detrend:
        for i in range(h):
            for j in range(w):
                # 对每个传感点的时间序列去除线性趋势
                data[:, i, j] = signal.detrend(data[:, i, j])

    # 3. 时间维度平滑（滑动平均，消除时间上的随机毛刺）
    if time_smooth_window > 1:
        # 构造滑动平均核
        kernel = np.ones(time_smooth_window) / time_smooth_window
        for i in range(h):
            for j in range(w):
                # 卷积实现滑动平均，保持长度不变
                data[:, i, j] = signal.convolve(data[:, i, j], kernel, mode='same')

    # 4. 空间维度去噪（中值滤波+高斯滤波，消除空间孤立噪声）
    # 先中值滤波（对脉冲噪声效果好）
    data = median_filter(data, size=(1, spatial_median_size, spatial_median_size))
    # 再高斯滤波（平滑空间分布）
    data = gaussian_filter(data, sigma=(0, gaussian_sigma, gaussian_sigma))

    return data


def analyze_sensor_correlation(data):
    """
    分析140个传感点的相关性
    参数:
        data: 去噪后的数据，形状为 (num_frame, 14, 10)
    返回:
        corr_matrix: 140×140的相关性矩阵
        corr_stats: 相关性统计信息（均值、最大值、最小值、高相关对）
    """
    # 检查输入数据形状是否符合要求
    if len(data.shape) != 3 or data.shape[1:] != (14, 10):
        raise ValueError(f"输入数据形状错误，期望 (num_frame, 14, 10)，实际 {data.shape}")

    num_frame, h, w = data.shape

    # 1. 将每个帧的14×10矩阵展平为140维向量
    flattened_data = data.reshape(num_frame, h * w)  # 形状变为 (num_frame, 140)

    # 2. 计算皮尔逊相关系数矩阵（140×140）
    corr_matrix = np.corrcoef(flattened_data.T)  # 转置后计算列之间的相关性

    # 3. 相关性统计分析
    # 修复核心：创建布尔类型的掩码，排除对角线（自身与自身的相关性=1）
    mask = np.ones_like(corr_matrix, dtype=bool)  # 显式指定布尔类型
    np.fill_diagonal(mask, False)  # 对角线设为False，排除自身相关性
    corr_vals = corr_matrix[mask]  # 直接用布尔掩码筛选，无需astype转换

    # 筛选高相关（>0.8）的点对（排除自身）
    high_corr_mask = (corr_matrix > 0.8) & mask
    high_corr_indices = np.where(high_corr_mask)
    # 将索引对转为更易读的格式（避免重复，如(0,1)和(1,0)只保留一个）
    high_corr_pairs = []
    for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
        if i < j:  # 只保留i<j的对，避免重复统计
            high_corr_pairs.append((i, j))

    # 统计信息
    corr_stats = {
        'mean_corr': np.mean(corr_vals),  # 整体平均相关性
        'max_corr': np.max(corr_vals),  # 最大相关性
        'min_corr': np.min(corr_vals),  # 最小相关性
        'high_corr_pairs_count': len(high_corr_pairs),  # 高相关对数量
        'high_corr_pairs': high_corr_pairs  # 高相关点对（索引）
    }

    # 4. 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='corr')
    plt.xlabel('0-139')
    plt.ylabel('0-139')
    plt.title('140 corr matrix')
    plt.tight_layout()
    plt.show()

    # 5. 可视化高相关区域（还原14×10空间分布）
    # 计算每个点的平均相关性（与其他所有点的相关性均值）
    avg_corr_per_point = np.mean(corr_matrix, axis=1).reshape(h, w)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(avg_corr_per_point, cmap='viridis')
    plt.colorbar(im, label='mean corr')
    plt.xlabel('Cols 0-9')
    plt.ylabel('Rows 0-13')
    plt.title('mean corr matrix')
    plt.tight_layout()
    plt.show()

    return corr_matrix, corr_stats

# ---------------------- 测试示例 ----------------------

raw_data = np.load("D:/blender_workplace/modules/PalmPoints_and_FingerBend/pressure_data_sequence.npy")

# print(raw_data.shape)

for i in [0, 1, 2, 3, 11, 12, 13]:
    for j in [6, 7, 8, 9]:
        plt.plot(raw_data[:, i, j], label=f'{i}, {j}', alpha=0.5)
plt.xlabel('time')
plt.ylabel('pressure')
plt.legend()
plt.show()

# # 去噪
# denoised_data = denoise_pressure_data(raw_data)
#
# # 可视化对比（第一个传感点的时间序列）
#
# plt.figure(figsize=(12, 4))
# plt.plot(raw_data[:, 4, 4], label='Raw Data', alpha=0.6)
# plt.plot(denoised_data[:, 4, 4], label='Denoised Data', linewidth=2)
# plt.xlabel('time')
# plt.ylabel('pressure')
# plt.legend()
# plt.show()