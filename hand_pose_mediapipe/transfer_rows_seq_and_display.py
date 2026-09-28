import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from IMUProcessing.MPU6050_utils import load_config, IMUData


def load_data_from_dataset(data):
    pressures = []
    angles = []

    print("Parsing dataset...")
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


def raw_data_to_true_img(list, pressure, max_row=13):
    pressure_real = np.zeros(pressure.shape)

    for i, row in enumerate(list):
        pressure_real[max_row - i] = pressure[row]

    return pressure_real


def cnn_img_to_raw_(list, pressure, max_row=13):
    pressure_real = np.zeros(pressure.shape)

    for i, row in enumerate(list):
        pressure_real[row] = pressure_array[max_row - i]

    return pressure_real



dataset = pd.read_csv("record/session_20260316_133505/dataset.csv")
pressure_data, angles_data = load_data_from_dataset(dataset)

# locate the coords of pressure sensor
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

idx = 10
pressure_array = pressure_data[idx]
pressure_array_flatten = pressure_array.copy().reshape(140, -1)

n_rows = pressure_array.shape[0]
n_cols = pressure_array.shape[1]
TOTAL_NUM = n_cols * n_rows
MAX_VALUE = 3
MIN_VALUE = -3

row_idx_list = [0, 13, 1, 12, 2, 11, 3, 10, 4, 9, 5, 8, 6, 7]

# print(pressure_array)
pressure_array_trans = cnn_img_to_raw_(row_idx_list, pressure_array)
pressure_array_trans_flatten = pressure_array_trans.copy().reshape(140, -1)
# print(pressure_array_trans)




fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.title("Real-time Palm Pressure Map ")

j_coords, i_coords = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

scatter = ax.scatter(
    j_coords,
    i_coords,
    c = pressure_array_flatten,
    # c = pressure_array_trans.copy().reshape(140, -1),  # 每个点的压力值（颜色）
    cmap = 'coolwarm',
    vmin = MIN_VALUE,
    vmax = MAX_VALUE,
    marker = 's',
    s = 800,  # 点的大小（可根据坐标调整）
    edgecolors = 'white',  # 点的边框（区分相邻点）
    linewidth = 0.5,
)

# 添加侧边颜色条
plt.colorbar(scatter, label='Pressure Intensity')

texts = []

for i in range(n_rows):
    for j in range(n_cols):
        x = j_coords[i, j]
        y = i_coords[i, j]
        text_content = f"{pressure_array[i, j]:.2f}"
        text = ax.text(x, y, text_content, ha="center", va="center",
                       fontsize=6, fontweight='bold')
        texts.append(text)

ax.set_title("Pressure Matrix Values")
ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))


scatter = ax2.scatter(
    [coord[0] for coord in sensor_coords],  # 所有点的x坐标
    [-coord[1] for coord in sensor_coords],
    c=pressure_array_trans_flatten,                         # 每个点的压力值（颜色）
    cmap='coolwarm',
    vmin=MIN_VALUE,
    vmax=MAX_VALUE,
    marker='s',
    s=200,                                  # 点的大小（可根据坐标调整）
    edgecolors='white',                     # 点的边框（区分相邻点）
    linewidth=0.5,
)
# 添加侧边颜色条
plt.colorbar(scatter, label='Pressure Intensity')

texts = []
for i, sensor_coord in enumerate(sensor_coords):
    x, y = sensor_coord
    text = ax2.text(x, -y, f"{pressure_array_trans_flatten[i][0]:.2f}", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)

ax2.set_title("Real-time Pressure Matrix Values")
ax2.set_xticks(np.arange(n_cols))
ax2.set_yticks(np.arange(n_rows))










plt.show()














