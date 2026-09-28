import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import serial
import threading
import queue
import time
import os
from PIL import Image
from identify_gesture_CNN import load_model, predict_single_image



ROWS, COLS = 14, 10             # 矩阵维度
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100           # 更新频率 (毫秒)，100ms = 10帧/秒
MAX_VALUE = 500
PREDICT_INTERVAL = 500          # 模型预测间隔(毫秒)，避免每帧预测
GESTURE_IMG_DIR = "Hand Gestures Pic/"  # 手势图片文件夹路径
CNN_MODEL_PATH = "trained_cnn_model.pth"

# 用于存储最新的预测结果（线程安全）
prediction_queue = queue.Queue(maxsize=1)
# 存储最新的手势图片
latest_gesture_img = None
# 控制预测线程的停止信号
predict_stop_event = threading.Event()

model = load_model(CNN_MODEL_PATH, num_classes=11)


def predict_worker():
    """独立线程处理CNN预测，避免阻塞UI"""
    last_predict_time = 0
    while not predict_stop_event.is_set():
        current_time = time.time() * 1000

        # 控制预测频率
        if (current_time - last_predict_time) > PREDICT_INTERVAL:
            try:
                # 获取最新的压力数据
                if hasattr(threading_collect.palm_points, 'latest_pressure_data') and \
                        threading_collect.palm_points.latest_pressure_data is not None:
                    pressure_data = threading_collect.palm_points.latest_pressure_data.reshape(ROWS, COLS)

                    # 执行CNN预测
                    pred_label, pred_probs = predict_single_image(model, pressure_data)
                    # print('pred_label:', pred_label)

                    # 更新预测结果（队列只保留最新的）
                    if not prediction_queue.empty():
                        try:
                            prediction_queue.get_nowait()
                        except queue.Empty:
                            pass
                    prediction_queue.put(pred_label)

                    # 加载对应的手势图片
                    global latest_gesture_img
                    img_path = os.path.join(GESTURE_IMG_DIR, f"gesture_{pred_label + 1}.jpg")
                    if os.path.exists(img_path):
                        latest_gesture_img = Image.open(img_path)
                    else:
                        # 加载默认图片
                        latest_gesture_img = None

                last_predict_time = current_time
            except Exception as e:
                print(f"预测出错: {e}")

        # 短暂休眠，减少CPU占用
        time.sleep(0.01)

# locate the coords of pressure sensor
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

# init canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Real-time Palm Pressure & Gesture Recognition", fontsize=16)


initial_data = np.zeros(TOTAL_NUM)
ax1.set_title("Real-time Palm Pressure Map")

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

ax1.set_xticks(np.arange(COLS))
ax1.set_yticks(np.arange(ROWS))


ax2.set_title("Recognized Hand Gesture")
ax2.axis('off')  # 隐藏坐标轴
# 初始化默认图片
default_img = np.ones((200, 200, 3), dtype=np.uint8) * 220  # 浅灰色背景
gesture_img_display = ax2.imshow(default_img)


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

    global latest_gesture_img
    if latest_gesture_img is not None:
        # 将PIL图片转换为numpy数组
        img_array = np.array(latest_gesture_img)
        gesture_img_display.set_data(img_array)
        ax2.set_title(f"Recognized Gesture (Updated: {frame})")
    else:
        # 显示默认背景
        gesture_img_display.set_data(default_img)
        ax2.set_title("No Gesture Recognized")

    # 自动调整图片显示范围
    ax2.relim()
    ax2.autoscale_view()

    return scatter, gesture_img_display, *texts


imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")

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


# 启动数据采集线程
print('Starting data collection thread...')
threading_collect.t1 = threading.Thread(target=threading_collect.collecting, daemon=True)
threading_collect.t1.start()

# 启动CNN预测线程
print('Starting CNN prediction thread...')
predict_thread = threading.Thread(target=predict_worker, daemon=True)
predict_thread.start()


# 启动实时动画
# cache_frame_data=False 防止内存泄漏
print('Starting UI animation...')
ani = FuncAnimation(
    fig,
    update,
    interval=UPDATE_INTERVAL,
    blit=True,
    cache_frame_data=False
)


def on_close(event):
    """关闭窗口时的清理操作"""
    print('Stopping threads...')
    predict_stop_event.set()
    threading_collect.stop_event.set()
    ser.close()
    predict_thread.join(timeout=1)
    threading_collect.t1.join(timeout=1)
    print('All threads stopped.')

fig.canvas.mpl_connect('close_event', on_close)

plt.show()



