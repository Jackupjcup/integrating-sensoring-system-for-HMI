import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass




# ====================== 配置参数 ======================
ROWS, COLS = 14, 10
TOTAL_NUM = ROWS * COLS
MAX_VALUE = 100
MIN_VALUE = -100

# 初始化手掌点配置
palm_points = PalmPointsClass(ROWS, COLS, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

# ====================== 加载数据集 + 选择一帧数据 ======================
pressure_data = pd.read_csv(r'D:\blender_workplace\modules\Functions\objects_recognition\Objects_Collection\20260329-11\object_7\20260329_114433.csv')

# 选择要显示的帧（默认第 0 帧，你可以改成 1/2/...）
SHOW_FRAME = 0
one_frame_data = np.array(pressure_data.iloc[SHOW_FRAME])  # 形状 (140,)

# ====================== 创建画布布局 ======================
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111)

# ====================== 绘制左侧压力热力图 ======================
scatter = ax1.scatter(
    [coord[0] for coord in palm_points.plane_pressure_points_coords],
    [-coord[1] for coord in palm_points.plane_pressure_points_coords],
    c=one_frame_data,
    cmap='coolwarm',
    vmin=MIN_VALUE,
    vmax=MAX_VALUE,
    marker='s',
    s=200,
    edgecolors='white',
    linewidth=0.5,
)
plt.colorbar(scatter, ax=ax1, label='Pressure Intensity')

# 添加数值文字
texts = []
for idx, sensor_coord in enumerate(palm_points.plane_pressure_points_coords):
    x, y = sensor_coord
    val = one_frame_data[idx]
    text = ax1.text(x, -y, f"{val:.1f}", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    # 文字颜色自适应
    if val > MAX_VALUE * 0.6:
        text.set_color("white")
    texts.append(text)


# ====================== 显示图像 ======================
plt.suptitle(f"Pressure Heatmap - Frame {SHOW_FRAME}", fontsize=14)
plt.tight_layout()
plt.show()