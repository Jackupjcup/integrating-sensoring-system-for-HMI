"""
实时 IMU 姿态可视化

基于 ThreadingCollect_class.py 进行数据采集和滤波，
本文件只负责读取已滤波的数据并可视化。

使用方法：
    python visualize_imu_attitude.py

依赖：
    pip install matplotlib numpy pyserial
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import time

# 添加模块路径
sys.path.append("D:/blender_workplace/modules")
from ThreadingCollect_class import ThreadingCollect, imu_config, GRAVITY, DEG_TO_RAD, imu_lock
from IMUProcessing.MPU6050_utils import load_config, IMUData, quat_to_rot, euler_to_quat
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass

def draw_coordinate_frame(ax, R, origin=[0, 0, 0], scale=1.0, labels=['X', 'Y', 'Z']):
    """在 3D 坐标系中绘制坐标轴"""
    colors = ['red', 'green', 'blue']
    for i in range(3):
        direction = R[:, i] * scale
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=colors[i], arrow_length_ratio=0.1, linewidth=2
        )
        end_point = np.array(origin) + direction * 1.15
        ax.text(end_point[0], end_point[1], end_point[2], labels[i], color=colors[i], fontsize=12)


def create_visualization(imu_id=1, tc=None):
    """
    创建可视化界面。

    :param imu_id: IMU 编号 (1, 2, 3)
    :param tc: ThreadingCollect 实例
    """
    imu = tc.imu_dic[f'imu_{imu_id}']

    # 历史数据
    max_history = 200
    time_history = []
    roll_history = []
    pitch_history = []
    yaw_history = []
    start_time = time.time()

    # 创建图形 - 统一 2x3 布局
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f'IMU{imu_id} Real-time Attitude (Kalman Filtered)', fontsize=14, fontweight='bold')

    # 3D 姿态显示 (占 2 列)
    ax_3d = fig.add_subplot(2, 3, (1, 2), projection='3d')

    # 角度仪表盘 (占 1 列)
    ax_gauge = fig.add_subplot(2, 3, 3)

    # Roll / Pitch / Yaw 时序图
    ax_roll = fig.add_subplot(2, 3, 4)
    ax_pitch = fig.add_subplot(2, 3, 5)
    ax_yaw = fig.add_subplot(2, 3, 6)

    plt.tight_layout()

    def update(frame):
        # 线程安全读取 ThreadingCollect 中已滤波的数据
        with imu_lock:
            # angles = imu.get_kalman_angles()  # [Roll, Pitch, Yaw] deg
            angles = imu.get_raw_euler_degrees()
            quat = imu.Quat.copy()            # [w, x, y, z]
            # quat = euler_to_quat(*angles)
            roll, pitch, yaw = angles

            # 记录历史
            now = time.time()
            time_history.append(now - start_time)
            roll_history.append(roll)
            pitch_history.append(pitch)
            yaw_history.append(yaw)
            if len(time_history) > max_history:
                time_history.pop(0)
                roll_history.pop(0)
                pitch_history.pop(0)
                yaw_history.pop(0)

            # 复制用于绘图 (离开锁后再用)
            t_plot = list(time_history)
            r_plot = list(roll_history)
            p_plot = list(pitch_history)
            y_plot = list(yaw_history)

        # === 3D 姿态 ===
        ax_3d.cla()
        ax_3d.set_title('3D Attitude')
        ax_3d.set_xlim([-1.5, 1.5])
        ax_3d.set_ylim([-1.5, 1.5])
        ax_3d.set_zlim([-1.5, 1.5])
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')

        R = quat_to_rot(quat)
        draw_coordinate_frame(ax_3d, R, scale=1.0)

        # 参考坐标系 (灰色)
        ax_3d.quiver(0, 0, 0, 0.5, 0, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1, linestyle='--')
        ax_3d.quiver(0, 0, 0, 0, 0.5, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1, linestyle='--')
        ax_3d.quiver(0, 0, 0, 0, 0, 0.5, color='gray', alpha=0.3, arrow_length_ratio=0.1, linestyle='--')

        # === 角度数值 ===
        ax_gauge.cla()
        ax_gauge.set_title('Angle Values (deg)')
        ax_gauge.axis('off')
        angle_text = (
            f"  Roll:  {roll:+8.2f} deg\n"
            f"  Pitch: {pitch:+8.2f} deg\n"
            f"  Yaw:   {yaw:+8.2f} deg\n"
            f"\n"
            f"Quaternion:\n"
            f"  w: {quat[0]:+.4f}\n"
            f"  x: {quat[1]:+.4f}\n"
            f"  y: {quat[2]:+.4f}\n"
            f"  z: {quat[3]:+.4f}"
        )
        ax_gauge.text(0.5, 0.5, angle_text, transform=ax_gauge.transAxes,
                      fontsize=12, verticalalignment='center', horizontalalignment='center',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # === 时序图 ===
        if len(t_plot) > 1:
            for ax, data, color, title in [
                (ax_roll, r_plot, 'r', 'Roll'),
                (ax_pitch, p_plot, 'g', 'Pitch'),
                (ax_yaw, y_plot, 'b', 'Yaw (drift expected)'),
            ]:
                ax.cla()
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angle (deg)')
                if title != 'Yaw (drift expected)':
                    ax.set_ylim([-180, 180])
                ax.grid(True, alpha=0.3)
                ax.plot(t_plot, data, color + '-', linewidth=1.5)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        return []

    anim = FuncAnimation(fig, update, interval=5, blit=False, cache_frame_data=False)

    def on_close(event):
        tc.stop_collect()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import serial

    print("=" * 50)
    print("IMU Real-time Attitude Visualization")
    print("=" * 50)
    print("\nAvailable IMUs: 1, 2, 3")
    print("Enter IMU ID to visualize (default=1):")

    try:
        user_input = 2#input().strip()
        imu_id = int(user_input) if user_input else 1
        if imu_id not in [1, 2, 3]:
            print("Invalid IMU ID, using default (1)")
            imu_id = 1
    except:
        imu_id = 1

    # === 初始化串口 ===
    ser = serial.Serial(
        port=imu_config['serial']['port'],
        baudrate=imu_config['serial']['baud_rate'],
        timeout=imu_config['serial']['timeout']
    )

    # === 初始化 IMU ===
    imu_config_path = "D:/blender_workplace/modules/IMUProcessing/config.json"
    imu_1 = IMUData('mpu6050_1', imu_config_path)
    imu_2 = IMUData('mpu6050_2', imu_config_path)
    imu_3 = IMUData('mpu6050_3', imu_config_path)

    # === 初始化 PalmPoints (ThreadingCollect 需要) ===
    palm_points = PalmPointsClass(14, 10, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

    # === 创建 ThreadingCollect ===
    tc = ThreadingCollect(ser=ser, imu_1=imu_1, imu_2=imu_2, imu_3=imu_3,
                          palm_points=palm_points, duration=5000)

    # === 启动采集 ===
    tc.is_collecting = True
    tc.stop_event.clear()
    tc.Start_time = time.time()
    tc.t1 = __import__('threading').Thread(target=tc.collecting, daemon=True)
    tc.t1.start()

    print(f"\nStarting visualization for IMU{imu_id}...")
    print("Close the window to stop.\n")

    # === 启动可视化 (阻塞) ===
    create_visualization(imu_id=imu_id, tc=tc)
