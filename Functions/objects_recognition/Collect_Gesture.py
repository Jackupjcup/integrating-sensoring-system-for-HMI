import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from matplotlib.animation import FuncAnimation
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading
import datetime


CSV_FILE = 'pressure_data.csv'  # 你的CSV文件名
ROWS, COLS = 14, 10             # 矩阵维度
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 1000




# locate the coords of pressure sensor
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

# init canvas
fig = plt.figure(figsize=(12, 6))  # 设置画布大小

# 使用gridspec划分布局：2行3列
# 左侧大图占2行2列（行0-1，列0-1），右侧两个小图各占1行1列（行0列2、行1列2）
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[2, 2, 1, 1])

# 创建三个轴对象
ax1 = fig.add_subplot(gs[:, :2])  # 左侧大图的轴（所有行，前两列）
ax2 = fig.add_subplot(gs[0, 2:])  # 右侧上图的轴（第0行，第2列）
ax3 = fig.add_subplot(gs[1, 2:])  # 右侧下图的轴（第1行，第2列）

ax2.set_xlim(-1, 10)
ax2.set_ylim(-500, 500)  # 扩大y轴范围，适配多条折线
ax3.set_xlim(-1, 14)
ax3.set_ylim(-500, 500)
initial_data = np.zeros(TOTAL_NUM)
title_text = ax1.set_title("Real-time Palm Pressure Map (Press 'SPACE' to Record)", fontsize=14)
# plt.title("Real-time Palm Pressure Map ")

scatter = ax1.scatter(
    [coord[0] for coord in sensor_coords],  # 所有点的x坐标
    [-coord[1] for coord in sensor_coords],
    c=initial_data,                         # 每个点的压力值（颜色）
    cmap='coolwarm',
    vmin=0,
    vmax=MAX_VALUE,
    marker='s',
    s=200,                                  # 点的大小（可根据坐标调整）
    edgecolors='white',                     # 点的边框（区分相邻点）
    linewidth=0.5,
)

# 添加侧边颜色条
plt.colorbar(scatter, label='Pressure Intensity')

texts = []
for sensor_coord in sensor_coords:
    x, y = sensor_coord
    text = ax1.text(x, -y, "", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)


# ax1.set_xticks(np.arange(COLS))
# ax1.set_yticks(np.arange(ROWS))

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


def save_sequence_to_csv(data_list):
    """
    将缓存的数据保存到 CSV
    data_list: 包含10个numpy数组的列表
    """
    # 1. 堆叠数组，变成 (10, 140) 的形状
    try:
        np_data = np.vstack(data_list)
    except Exception as e:
        print(f"[Error] 数据堆叠失败: {e}")
        return

    # 2. 生成基于当前时间的文件夹路径
    # 格式: current_datetime (例如 20231027_153022)
    hand_gesture_dir = "Objects_Collection"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.getcwd(), hand_gesture_dir, LOG_DATE_TIME, GESTURE_NAME)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. 完整文件路径
    # file_path = os.path.join(save_dir, SAVE_FILENAME)
    file_name = current_datetime + ".csv"
    file_path = os.path.join(save_dir, file_name)
    # 4. 保存
    try:
        # 使用逗号分隔保存，保留2位小数
        np.savetxt(file_path, np_data, delimiter=",", fmt="%.2f")
        print(f"\n[保存成功{CLICK_COUNT}个] 路径: {file_path}")
        title_text.set_text(f"Saved: {current_datetime}/{GESTURE_NAME}, Click: {CLICK_COUNT}")
        title_text.set_color("green")  # 变绿提示成功
    except Exception as e:
        print(f"[保存失败] {e}")

def on_key_press(event):
    """按键事件处理"""
    global is_recording, frames_buffer, CLICK_COUNT

    # 按下 ' ' (空格) 或 'r' 键开始
    if event.key in [' ', 'r']:
        if not is_recording:
            is_recording = True
            CLICK_COUNT += 1
            frames_buffer = []  # 清空缓存
            print("\n>>> 开始录制 10 组数据...")
            title_text.set_text(">>> RECORDING DATA... <<<")
            title_text.set_color("red")  # 变红提示正在录制
        else:
            print("正在录制中，请勿重复按键...")

fig.canvas.mpl_connect('key_press_event', on_key_press)

def update(frame):
    global is_recording, frames_buffer
    """动画更新函数"""
    with data_lock:
        # 获取最新压力数据
        if hasattr(threading_collect, 'palm_points') and threading_collect.palm_points.latest_pressure_data is not None:
            data = threading_collect.palm_points.latest_pressure_data
        else:
            data = np.zeros(TOTAL_NUM)  # 无数据时显示0

    if is_recording:
        # 必须使用 .copy()，否则存的是引用，数据会变
        frames_buffer.append(data.copy())

        # 检查是否存满 10 帧
        if len(frames_buffer) >= TARGET_FRAMES:
            # 停止录制标志位
            is_recording = False
            # 执行保存 (可以放在这里，也可以另开线程，这里数据量小直接存即可)
            save_sequence_to_csv(frames_buffer)
            # 稍后恢复标题颜色
            plt.pause(0.1)

    # data[np.where(data < 0)] = 0
    scatter.set_array(data) # 更新矩阵数据

    # 遍历每个格子，更新文字
    for idx in range(TOTAL_NUM):
        val = data[idx]
        texts[idx].set_text(f"{int(data[idx])}")

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

    # 必须返回更新的图像对象（FuncAnimation要求返回可迭代的对象）
    return scatter, *texts, *rows_lines, *cols_lines

########################################################################################################################
########################################################################################################################

imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

# 配置串口
ser = serial.Serial(
    port=imu_config['serial']['port'],  # 根据实际情况修改端口号
    baudrate=imu_config['serial']['baud_rate'],  # 根据实际情况修改波特率
    timeout=imu_config['serial']['timeout']
)


# Initiate IMU Params
imu_config_path = "D:/blender_workplace/modules/IMUProcessing/config.json"
imu_1 = IMUData('mpu6050_1', imu_config_path)
imu_2 = IMUData('mpu6050_2', imu_config_path)
imu_3 = IMUData('mpu6050_3', imu_config_path)

# Initiate Palm Points
palm_points = PalmPointsClass(ROWS, COLS, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")


threading_collect = ThreadingCollect(ser=ser,
                                     imu_1=imu_1,
                                     imu_2=imu_2,
                                     imu_3=imu_3,
                                     palm_points=palm_points,
                                     duration=500)

# threading_collect.pressure_sensor_calibration(pressure_calibration_duration=10)
# palm_points.press_sensor_zero_bias_np = np.zeros(TOTAL_NUM)

# --- 录制相关变量 ---
is_recording = False       # 是否正在录制
frames_buffer = []         # 缓存数据的列表
TARGET_FRAMES = 100         # 目标录制帧数 (10帧)
CLICK_COUNT = 0

LOG_DATE_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")[: 11]
GESTURE_NAME = "object_7"

threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()
palm_points.calibration_flag = False

print('The main thread is starting')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

# 启动实时动画
# cache_frame_data=False 防止内存泄漏
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False)

plt.show()


