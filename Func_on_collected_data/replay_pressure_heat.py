import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading


def load_dataset(data):
    pressures = []
    angles = []

    for idx in range(len(data)):
        # 解析pressure
        pressure_str = data.iloc[idx]['pressure']
        # 移除所有括号和换行符，用空格替换
        pressure_str = pressure_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
        # 移除多余空格，分割字符串
        pressure_array = np.fromstring(pressure_str, sep=' ', dtype=np.float32)
        pressure_array = pressure_array.reshape(14, 10)

        # 解析angles
        angles_str = data.iloc[idx]['angles']
        angles_str = angles_str.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
        angles_array = np.fromstring(angles_str, sep=' ', dtype=np.float32)

        pressures.append(pressure_array)
        angles.append(angles_array)

    # 转换为numpy数组
    pressures = np.array(pressures)
    angles = np.array(angles)

    return pressures, angles



ROWS, COLS = 14, 10             # 矩阵维度
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 1
MIN_VALUE = -1

initial_data = np.zeros(TOTAL_NUM)



# Initiate Palm Points
palm_points = PalmPointsClass(ROWS, COLS, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")



# ---------------------- 1. 创建画布和布局 ----------------------
fig = plt.figure(figsize=(12, 6))  # 设置画布大小

# 使用gridspec划分布局：2行3列
# 左侧大图占2行2列（行0-1，列0-1），右侧两个小图各占1行1列（行0列2、行1列2）
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[2, 2, 1, 1])

# 创建三个轴对象
ax1 = fig.add_subplot(gs[:, :2])  # 左侧大图的轴（所有行，前两列）
ax2 = fig.add_subplot(gs[0, 2:])  # 右侧上图的轴（第0行，第2列）
ax3 = fig.add_subplot(gs[1, 2:])  # 右侧下图的轴（第1行，第2列）

ax2.set_xlim(-1, 10)
ax2.set_ylim(-5, 5)  # 扩大y轴范围，适配多条折线
ax3.set_xlim(-1, 14)
ax3.set_ylim(-5, 5)



# ---------------------- 2. 初始化图像对象 ----------------------
# 创建初始图像对象（返回的im对象用于后续更新数据）
scatter = ax1.scatter(
    [coord[0] for coord in palm_points.plane_pressure_points_coords],  # 所有点的x坐标
    [-coord[1] for coord in palm_points.plane_pressure_points_coords],
    c=initial_data,                         # 每个点的压力值（颜色）
    cmap='coolwarm',
    vmin=MIN_VALUE,
    vmax=MAX_VALUE,
    marker='s',
    s=200,                                  # 点的大小（可根据坐标调整）
    edgecolors='white',                     # 点的边框（区分相邻点）
    linewidth=0.5,
)
# 添加侧边颜色条
plt.colorbar(scatter, ax=ax1, label='Pressure Intensity')
texts = []
for sensor_coord in palm_points.plane_pressure_points_coords:
    x, y = sensor_coord
    text = ax1.text(x, -y, "", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)


init_row_data = np.zeros(COLS)
row_x = np.arange(len(init_row_data))
rows_lines = []
for i in range(ROWS):
    im2, = ax2.plot(row_x, init_row_data, label=f"row {i}")
    rows_lines.append(im2)
ax2.legend(fontsize=5)

init_col_data = np.zeros(ROWS)
col_x = np.arange(len(init_col_data))
cols_lines = []
for i in range(COLS):
    im3, = ax3.plot(col_x, init_col_data, label=f"col {i}")
    cols_lines.append(im3)
ax3.legend(fontsize=5)


data_lock = threading.Lock()

# ---------------------- 3. 定义动画更新函数 ----------------------
def update(frame):
    """
    动画更新函数，每次帧刷新时调用
    frame: 帧序号（由FuncAnimation自动传入）
    """

    """动画更新函数"""
    global COUNT

    if COUNT <= num_frame:
        data = pressure_data[COUNT].reshape(140, -1).squeeze()

        # 更新三个图像对象的数据（核心：只更新data，不重建对象）
        scatter.set_array(data)  # 更新矩阵数据

        for idx in range(TOTAL_NUM):
            val = data[idx]
            texts[idx].set_text(f"{data[idx]:.1f}")

            # 智能调整文字颜色：如果背景太红/太深，文字改为白色
            if val > MAX_VALUE * 0.6:
                texts[idx].set_color("white")
            else:
                texts[idx].set_color("black")

        reshaped_data = np.reshape(data, (ROWS, COLS))

        for i in range(ROWS):
            rows_lines[i].set_ydata(reshaped_data[i, :])

        for j in range(COLS):
            cols_lines[j].set_ydata(reshaped_data[:, j])

        COUNT += 1

        # 必须返回更新的图像对象（FuncAnimation要求返回可迭代的对象）
        return scatter,  *texts, *rows_lines, *cols_lines

    else:
        return []



COUNT = 0
dataset = pd.read_csv(r'D:\blender_workplace\modules\hand_pose_mediapipe\record\session_20260314_195645\dataset.csv')
pressure_data, _ = load_dataset(dataset)
num_frame = pressure_data.shape[0]
# print(pressure_data[0].reshape(140, -1).squeeze().shape)



# 创建动画对象
ani = FuncAnimation(
    fig=fig,  # 绑定的画布
    func=update,  # 帧更新函数
    frames=np.arange(0, 100),  # 帧范围（0到99，循环播放）
    interval=50,  # 帧间隔（毫秒），越小刷新越快
    blit=True,  # 开启blitting优化（提升刷新效率）
    repeat=True  # 循环播放
)

# 显示画布
# plt.tight_layout()  # 自动调整布局，避免重叠
plt.show()