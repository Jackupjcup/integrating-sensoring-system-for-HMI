import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading


CSV_FILE = 'pressure_data.csv'  # 你的CSV文件名
ROWS, COLS = 14, 10             # 矩阵维度import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading


CSV_FILE = 'pressure_data.csv'  # 你的CSV文件名
ROWS, COLS = 14, 10             # 矩阵维度
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 500

# locate the coords of pressure sensor
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

# init canvas
fig, ax = plt.subplots(figsize=(10, 10))
initial_data = np.zeros(TOTAL_NUM)
plt.title("Real-time Palm Pressure Map ")

scatter = ax.scatter(
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
    text = ax.text(x, -y, "", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)

ax.set_title("Real-time Pressure Matrix Values")
ax.set_xticks(np.arange(COLS))
ax.set_yticks(np.arange(ROWS))




def update(frame):
    global data
    """动画更新函数"""
    if threading_collect.palm_points.latest_pressure_data is not None:
        data = threading_collect.palm_points.latest_pressure_data#.reshape(ROWS, COLS)
    else:
        data = np.zeros(TOTAL_NUM)

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

    return scatter, *texts


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

threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()

print('The main thread is starting')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

# 启动实时动画
# cache_frame_data=False 防止内存泄漏
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False)

plt.show()



TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 1000

# locate the coords of pressure sensor
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

# init canvas
fig, ax = plt.subplots(figsize=(10, 10))
initial_data = np.zeros(TOTAL_NUM)
plt.title("Real-time Palm Pressure Map ")

scatter = ax.scatter(
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
    text = ax.text(x, -y, "", ha="center", va="center",
                   fontsize=6, fontweight='bold')
    texts.append(text)

ax.set_title("Real-time Pressure Matrix Values")
ax.set_xticks(np.arange(COLS))
ax.set_yticks(np.arange(ROWS))




def update(frame):
    global data
    """动画更新函数"""
    if threading_collect.palm_points.latest_pressure_data is not None:
        data = threading_collect.palm_points.latest_pressure_data#.reshape(ROWS, COLS)
    else:
        data = np.zeros(TOTAL_NUM)

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

    return scatter, *texts


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

threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()

print('The main thread is starting')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

# 启动实时动画
# cache_frame_data=False 防止内存泄漏
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False)

plt.show()


