import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading


ROWS, COLS = 14, 10             # 矩阵维度
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 500

initial_data = np.zeros(TOTAL_NUM)


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

# ---------------------- 1. 创建画布和布局 ----------------------
fig = plt.figure(figsize=(12, 6))  # 设置画布大小

# 使用gridspec划分布局：2行3列
# 左侧大图占2行2列（行0-1，列0-1），右侧两个小图各占1行1列（行0列2、行1列2）
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

# 创建三个轴对象
ax1 = fig.add_subplot(gs[0:, :2])
ax2 = fig.add_subplot(gs[0:, 2:])



ax1.set_xlim(-1, 10)
ax1.set_ylim(-1000, 1000)  # 扩大y轴范围，适配多条折线
ax2.set_xlim(-1, 10)
ax2.set_ylim(-1000, 1000)  # 扩大y轴范围，适配多条折线


# ---------------------- 2. 初始化图像对象 ----------------------

init_row_data = np.zeros(COLS)
row_x = np.arange(len(init_row_data))
rows_lines = []
for i in range(ROWS):
    im1, = ax1.plot(row_x, init_row_data, label=f"row {i}")
    rows_lines.append(im1)
ax1.legend(fontsize=5)
ax1.title.set_text("Denoised data")

init_raw_row_data = np.zeros(COLS)
raw_row_x = np.arange(len(init_raw_row_data))
raw_rows_lines = []
for i in range(ROWS):
    im2, = ax2.plot(raw_row_x, init_raw_row_data, label=f"Raw row {i}")
    raw_rows_lines.append(im2)
ax2.legend(fontsize=5)
ax2.title.set_text("Initial data")


data_lock = threading.Lock()

# ---------------------- 3. 定义动画更新函数 ----------------------
def update(frame):
    """
    动画更新函数，每次帧刷新时调用
    frame: 帧序号（由FuncAnimation自动传入）
    """

    """动画更新函数"""
    with data_lock:
        # 获取最新压力数据
        if hasattr(threading_collect, 'palm_points') and threading_collect.palm_points.latest_pressure_data is not None:
            data = threading_collect.palm_points.latest_pressure_data
            raw_data = threading_collect.palm_points.raw_pressure_data
        else:
            data = np.zeros(TOTAL_NUM)  # 无数据时显示0
            raw_data = np.zeros(TOTAL_NUM)


    reshaped_data = np.reshape(data, (ROWS, COLS))
    reshaped_raw_data = np.reshape(raw_data, (ROWS, COLS))

    for i in range(ROWS):
        rows_lines[i].set_ydata(reshaped_data[i, :])

    for j in range(ROWS):
        raw_rows_lines[j].set_ydata(reshaped_raw_data[j, :])

    # 必须返回更新的图像对象（FuncAnimation要求返回可迭代的对象）
    return *rows_lines, *raw_rows_lines





# threading_collect.pressure_sensor_calibration(pressure_calibration_duration=10)
# palm_points.press_sensor_zero_bias_np = np.zeros(TOTAL_NUM)

threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()

print('The main thread is starting')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

# ---------------------- 4. 启动动画 ----------------------
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