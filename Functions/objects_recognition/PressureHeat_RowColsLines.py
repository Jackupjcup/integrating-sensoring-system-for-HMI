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
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[2, 2, 1, 1])

# 创建三个轴对象
ax1 = fig.add_subplot(gs[:, :2])  # 左侧大图的轴（所有行，前两列）
ax2 = fig.add_subplot(gs[0, 2:])  # 右侧上图的轴（第0行，第2列）
ax3 = fig.add_subplot(gs[1, 2:])  # 右侧下图的轴（第1行，第2列）

ax2.set_xlim(-1, 10)
ax2.set_ylim(-1000, 1000)  # 扩大y轴范围，适配多条折线
ax3.set_xlim(-1, 14)
ax3.set_ylim(-1000, 1000)



# ---------------------- 2. 初始化图像对象 ----------------------
# 创建初始图像对象（返回的im对象用于后续更新数据）
scatter = ax1.scatter(
    [coord[0] for coord in palm_points.plane_pressure_points_coords],  # 所有点的x坐标
    [-coord[1] for coord in palm_points.plane_pressure_points_coords],
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
    with data_lock:
        # 获取最新压力数据
        if hasattr(threading_collect, 'palm_points') and threading_collect.palm_points.latest_pressure_data is not None:
            data = threading_collect.palm_points.latest_pressure_data
        else:
            data = np.zeros(TOTAL_NUM)  # 无数据时显示0

    # 更新三个图像对象的数据（核心：只更新data，不重建对象）
    scatter.set_array(data) # 更新矩阵数据

    if threading_collect.palm_points.collected_points_df.shape[0] % 300 == 0:
        print(threading_collect.palm_points.collected_points_df.shape[0])

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





# threading_collect.pressure_sensor_calibration(pressure_calibration_duration=10)
# palm_points.press_sensor_zero_bias_np = np.zeros(TOTAL_NUM)

threading_collect.is_collecting = True
threading_collect.main_thread_flag = True
threading_collect.stop_event.clear()
palm_points.calibration_flag = False

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