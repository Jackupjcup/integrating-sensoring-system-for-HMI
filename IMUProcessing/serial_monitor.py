"""
串口监视器 - tkinter 版 (IMU 数据收集与可视化)
功能：
1. 串口连接/断开控制
2. IMU 数据收集 (基于 ThreadingCollect_class.py)
3. 实时 IMU 姿态可视化面板
4. 数据保存功能
"""

import sys
import json
import serial
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import deque

# 添加模块路径
sys.path.append("D:/blender_workplace/modules")
from IMUProcessing.MPU6050_utils import load_config, IMUData, quat_to_rot
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass

# ================= 配置 =================
CONFIG_PATH = "D:/blender_workplace/modules/IMUProcessing/config.json"
PALM_CONFIG_PATH = "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json"
MAX_HISTORY = 200  # 历史数据点数
UPDATE_INTERVAL = 50  # ms
# ========================================

# 全局常量
imu_config = load_config(CONFIG_PATH)
GRAVITY = imu_config['mpu6050_1']['Acc_params']['GRAVITY']
DEG_TO_RAD = np.pi / 180.0
imu_lock = threading.Lock()


class IMUCollector:
    """IMU 数据收集器 (基于 ThreadingCollect_class.py)"""

    def __init__(self, ser, imu_1, imu_2, imu_3):
        self.serial = ser
        self.imu_dic = {'imu_1': imu_1, 'imu_2': imu_2, 'imu_3': imu_3}
        self.COUNT_NUM = 0
        self.Start_time = time.time()

        self.is_collecting = False
        self.stop_event = threading.Event()
        self.IMU_is_capturing = False
        self.capture_X1_counts = 0
        self.collect_thread = None

        self.imu_dt_list = []
        self.imu_last_time = time.time()

    def start_collect(self):
        """启动数据收集线程"""
        self.is_collecting = True
        self.stop_event.clear()
        self.Start_time = time.time()
        self.COUNT_NUM = 0
        self.imu_dt_list = []
        self.imu_last_time = time.time()

        # 重置 IMU 数据
        for i in range(1, 4):
            self.imu_dic[f'imu_{i}'].reset()

        self.collect_thread = threading.Thread(target=self._collecting, daemon=True)
        self.collect_thread.start()

    def stop_collect(self):
        """停止数据收集"""
        if not self.is_collecting or self.collect_thread is None:
            return

        self.is_collecting = False
        self.stop_event.set()
        self.collect_thread.join(timeout=1)
        self.collect_thread = None

    def _collecting(self):
        """数据收集主循环"""
        import re
        pattern = re.compile(r'^([XYZ])(\d)A(-?\d+)G(-?\d+)$')

        while not self.stop_event.is_set():
            try:
                line_bytes = self.serial.readline()
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                match = pattern.match(line)
            except Exception:
                continue

            if not line:
                continue

            # IMU 数据解析
            if match and not self.IMU_is_capturing:
                axis, imu_id, acc, gyro = match.groups()
                if axis == "X" and imu_id == "1":
                    self.IMU_is_capturing = True
                    self.capture_X1_counts += 1

                    self.imu_dic['imu_1'].IMU_data_dic[f"Acc{axis}"], self.imu_dic['imu_1'].IMU_data_dic[f"Gyro{axis}"] = (
                        float(acc) / self.imu_dic['imu_1'].ACC_SENSITIVITY * GRAVITY - self.imu_dic['imu_1'].Acc_zero_bias[0],
                        float(gyro) / self.imu_dic['imu_1'].GYRO_SENSITIVITY * DEG_TO_RAD - self.imu_dic['imu_1'].Gyro_zero_bias[0])

            elif match and self.IMU_is_capturing:
                axis, imu_id, acc, gyro = match.groups()
                zero_bias_idx = 0 if axis == "X" else (1 if axis == "Y" else 2)

                self.imu_dic[f"imu_{imu_id}"].IMU_data_dic[f"Acc{axis}"], self.imu_dic[f"imu_{imu_id}"].IMU_data_dic[f"Gyro{axis}"] = (
                    float(acc) / self.imu_dic[f"imu_{imu_id}"].ACC_SENSITIVITY * GRAVITY - self.imu_dic[f"imu_{imu_id}"].Acc_zero_bias[zero_bias_idx],
                    float(gyro) / self.imu_dic[f"imu_{imu_id}"].GYRO_SENSITIVITY * DEG_TO_RAD - self.imu_dic[f'imu_{imu_id}'].Gyro_zero_bias[zero_bias_idx])

                # 一个完整的 IMU 数据帧 (X1 -> Z3)
                if axis == "Z" and imu_id == "3":
                    self.IMU_is_capturing = False

                    # 计算时间间隔
                    imu_now_time = time.time()
                    imu_dt = imu_now_time - self.imu_last_time
                    self.imu_last_time = imu_now_time
                    self.imu_dt_list.append(imu_dt)

                    with imu_lock:
                        for i in range(1, 4):
                            self.imu_dic[f"imu_{i}"].collect_imu_data()
                            self.imu_dic[f'imu_{i}'].kalman_update_angles(imu_dt)
                            self.imu_dic[f'imu_{i}'].gyro_to_quat(gyro=self.imu_dic[f'imu_{i}'].get_kalman_angles(), dt=imu_dt)
                            self.imu_dic[f'imu_{i}'].Quat_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].Quat
                            self.imu_dic[f'imu_{i}'].imu_data_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].imu_np
                            self.imu_dic[f"imu_{i}"].clear_dic()

                        self.COUNT_NUM += 1

    def get_imu_data(self, imu_id=1):
        """获取当前 IMU 数据"""
        with imu_lock:
            imu = self.imu_dic[f'imu_{imu_id}']
            angles = imu.get_kalman_angles()  # [Roll, Pitch, Yaw] deg
            quat = imu.Quat.copy()
            return angles, quat

    def get_count(self):
        """获取已收集的数据帧数"""
        return self.COUNT_NUM

    def save_data(self, save_dir):
        """保存 IMU 数据到 CSV"""
        for i in range(1, 4):
            imu = self.imu_dic[f'imu_{i}']
            imu.imu_data_df.to_csv(f"{save_dir}/imu_{i}_raw_data.csv", index=False)
            imu.Quat_df.to_csv(f"{save_dir}/imu_{i}_quat.csv", index=False)
        print(f"Saved {self.COUNT_NUM} frames to {save_dir}")


class SerialMonitor:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.is_connected = False

        # IMU 相关
        self.imu_config_path = CONFIG_PATH
        self.imu_1 = IMUData('mpu6050_1', self.imu_config_path)
        self.imu_2 = IMUData('mpu6050_2', self.imu_config_path)
        self.imu_3 = IMUData('mpu6050_3', self.imu_config_path)
        self.imu_collector = None

        # 当前选择的 IMU
        self.current_imu_id = 1

        # 历史数据
        self.time_history = deque(maxlen=MAX_HISTORY)
        self.roll_history = deque(maxlen=MAX_HISTORY)
        self.pitch_history = deque(maxlen=MAX_HISTORY)
        self.yaw_history = deque(maxlen=MAX_HISTORY)
        self.start_time = time.time()

        # 舵机目标角度历史 (用于对比)
        self.servo_target_angle = 0.0  # 当前目标角度
        self.servo_angle_history = deque(maxlen=MAX_HISTORY)  # 目标角度历史
        self.servo_imu_history = deque(maxlen=MAX_HISTORY)    # IMU测量角度历史
        self.servo_time_history = deque(maxlen=MAX_HISTORY)   # 时间历史

        # matplotlib 动画
        self.anim = None

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title(f"IMU 数据收集器 - {port} @ {baudrate}")
        self.root.geometry("1200x750")

        self.init_ui()

    def init_ui(self):
        # 主布局：左侧控制面板 + 右侧 IMU 数据面板
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ===== 左侧控制面板 =====
        panel = tk.LabelFrame(main_frame, text="控制", padx=10, pady=10)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # 连接按钮
        self.btn_connect = tk.Button(
            panel, text="连接", width=15, height=2,
            bg="#4CAF50", fg="white", font=("Arial", 12),
            command=self.toggle_connection
        )
        self.btn_connect.pack(pady=5)

        # 开始收集按钮 (初始禁用)
        self.btn_start_collect = tk.Button(
            panel, text="开始收集", width=15, height=2,
            bg="#2196F3", fg="white", font=("Arial", 12),
            state=tk.DISABLED, command=self.toggle_collect
        )
        self.btn_start_collect.pack(pady=5)

        # IMU 选择
        imu_frame = tk.LabelFrame(panel, text="IMU 选择", padx=5, pady=5)
        imu_frame.pack(fill=tk.X, pady=5)

        self.imu_var = tk.IntVar(value=1)
        for i in [1, 2, 3]:
            rb = tk.Radiobutton(
                imu_frame, text=f"IMU {i}", variable=self.imu_var,
                value=i, command=self.on_imu_select
            )
            rb.pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_label = tk.Label(
            panel, text=f"端口: {self.port}\n波特率: {self.baudrate}\n状态: 未连接",
            justify=tk.LEFT, font=("Arial", 10)
        )
        self.status_label.pack(pady=10)

        # 数据计数标签
        self.count_label = tk.Label(
            panel, text="已收集: 0 帧",
            font=("Arial", 10)
        )
        self.count_label.pack(pady=5)

        # ===== 舵机控制 =====
        servo_frame = tk.LabelFrame(panel, text="舵机角度控制", padx=5, pady=5)
        servo_frame.pack(fill=tk.X, pady=5)

        # 角度输入框
        self.servo_angle_var = tk.StringVar(value="0")
        servo_entry_frame = tk.Frame(servo_frame)
        servo_entry_frame.pack(fill=tk.X, pady=2)

        tk.Label(servo_entry_frame, text="角度:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.servo_entry = tk.Entry(servo_entry_frame, width=8, textvariable=self.servo_angle_var)
        self.servo_entry.pack(side=tk.LEFT, padx=5)
        self.servo_entry.bind("<Return>", lambda e: self.send_servo_angle())

        # 发送按钮
        self.btn_send_servo = tk.Button(
            servo_entry_frame, text="发送", width=6,
            state=tk.DISABLED, command=self.send_servo_angle
        )
        self.btn_send_servo.pack(side=tk.LEFT, padx=5)

        # 快捷角度按钮
        quick_angle_frame = tk.Frame(servo_frame)
        quick_angle_frame.pack(fill=tk.X, pady=2)

        for angle in [0, 30, 60, 90]:
            btn = tk.Button(
                quick_angle_frame, text=f"{angle}°", width=5,
                state=tk.DISABLED,
                command=lambda a=angle: self.send_quick_angle(a)
            )
            btn.pack(side=tk.LEFT, padx=2)
            setattr(self, f"btn_angle_{angle}", btn)

        # ===== 保存按钮 =====
        save_frame = tk.LabelFrame(panel, text="数据保存", padx=5, pady=5)
        save_frame.pack(fill=tk.X, pady=5)

        self.btn_save = tk.Button(
            save_frame, text="保存数据", width=12,
            state=tk.DISABLED, command=self.save_data
        )
        self.btn_save.pack(pady=5)

        # ===== 右侧 IMU 数据面板 =====
        imu_panel = tk.LabelFrame(main_frame, text="IMU 姿态数据", padx=5, pady=5)
        imu_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建 matplotlib 图形
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.suptitle('IMU Real-time Attitude', fontsize=12, fontweight='bold')

        # 3D 姿态显示
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 2), projection='3d')

        # 角度仪表盘
        self.ax_gauge = self.fig.add_subplot(2, 3, 3)

        # Roll / Pitch / Yaw 时序图
        self.ax_roll = self.fig.add_subplot(2, 3, 4)
        self.ax_pitch = self.fig.add_subplot(2, 3, 5)
        self.ax_yaw = self.fig.add_subplot(2, 3, 6)

        plt.tight_layout()

        # 嵌入 tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=imu_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始化图表显示
        self._init_plots()

    def _init_plots(self):
        """初始化空图表"""
        # 3D 姿态
        self.ax_3d.set_title('3D Attitude')
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([-1.5, 1.5])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')

        # 角度仪表盘
        self.ax_gauge.set_title('Angle Values (deg)')
        self.ax_gauge.axis('off')
        self.ax_gauge.text(0.5, 0.5, "等待数据...", transform=self.ax_gauge.transAxes,
                          fontsize=12, ha='center', va='center')

        # 时序图
        for ax, title in [(self.ax_roll, 'Roll'), (self.ax_pitch, 'Pitch'), (self.ax_yaw, 'Yaw')]:
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def toggle_connection(self):
        if not self.is_connected:
            # 连接
            try:
                self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
                self.is_connected = True

                # 创建 IMU 收集器
                self.imu_collector = IMUCollector(
                    self.serial_port, self.imu_1, self.imu_2, self.imu_3
                )

                # 更新 UI
                self.btn_connect.config(text="断开", bg="#f44336")
                self.status_label.config(text=f"端口: {self.port}\n波特率: {self.baudrate}\n状态: 已连接")
                self.btn_start_collect.config(state=tk.NORMAL)
                self.btn_send_servo.config(state=tk.NORMAL)
                for angle in [0, 30, 60, 90]:
                    getattr(self, f"btn_angle_{angle}").config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("连接失败", f"无法打开串口: {e}")
        else:
            # 断开
            self.is_connected = False

            # 停止收集 (如果正在收集)
            if self.imu_collector and self.imu_collector.is_collecting:
                self.imu_collector.stop_collect()
                self.btn_start_collect.config(text="开始收集", bg="#2196F3")

            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            # 停止动画
            if self.anim:
                self.anim.event_source.stop()
                self.anim = None

            # 更新 UI
            self.btn_connect.config(text="连接", bg="#4CAF50")
            self.status_label.config(text=f"端口: {self.port}\n波特率: {self.baudrate}\n状态: 已断开")
            self.btn_start_collect.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)

    def toggle_collect(self):
        """开始/停止数据收集"""
        if self.imu_collector and not self.imu_collector.is_collecting:
            # 开始收集
            self.imu_collector.start_collect()
            self.start_time = time.time()

            # 清空历史
            self.time_history.clear()
            self.roll_history.clear()
            self.pitch_history.clear()
            self.yaw_history.clear()

            # 启动动画
            self.anim = FuncAnimation(
                self.fig, self.update_plot,
                interval=UPDATE_INTERVAL, blit=False, cache_frame_data=False
            )
            self.canvas.draw()

            # 更新 UI
            self.btn_start_collect.config(text="停止收集", bg="#FF9800")
            self.btn_save.config(state=tk.DISABLED)

        elif self.imu_collector and self.imu_collector.is_collecting:
            # 停止收集
            self.imu_collector.stop_collect()

            # 停止动画
            if self.anim:
                self.anim.event_source.stop()
                self.anim = None

            # 更新 UI
            self.btn_start_collect.config(text="开始收集", bg="#2196F3")
            self.btn_save.config(state=tk.NORMAL)
            self.status_label.config(
                text=f"端口: {self.port}\n波特率: {self.baudrate}\n状态: 已收集 {self.imu_collector.get_count()} 帧"
            )

    def update_plot(self, frame):
        """更新 IMU 可视化"""
        if not self.imu_collector or not self.imu_collector.is_collecting:
            return []

        # 获取 IMU 数据
        angles, quat = self.imu_collector.get_imu_data(self.current_imu_id)
        roll, pitch, yaw = angles

        # 更新计数
        self.count_label.config(text=f"Have collected: {self.imu_collector.get_count()} Frames")

        # 记录历史
        now = time.time()
        self.time_history.append(now - self.start_time)
        self.roll_history.append(roll)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        # === 3D 姿态 ===
        self.ax_3d.cla()
        self.ax_3d.set_title(f'IMU{self.current_imu_id} 3D Attitude')
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([-1.5, 1.5])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')

        R = quat_to_rot(quat)
        self._draw_coordinate_frame(self.ax_3d, R, scale=1.0)

        # 参考坐标系
        self.ax_3d.quiver(0, 0, 0, 0.5, 0, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1)
        self.ax_3d.quiver(0, 0, 0, 0, 0.5, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1)
        self.ax_3d.quiver(0, 0, 0, 0, 0, 0.5, color='gray', alpha=0.3, arrow_length_ratio=0.1)

        # === 角度仪表盘 (含舵机对比) ===
        self.ax_gauge.cla()
        self.ax_gauge.set_title('Angle Values & Servo Comparison')
        self.ax_gauge.axis('off')

        # 计算误差
        servo_error = roll - self.servo_target_angle

        angle_text = (
            f"  Roll:  {roll:+8.2f} deg\n"
            f"  Pitch: {pitch:+8.2f} deg\n"
            f"  Yaw:   {yaw:+8.2f} deg\n"
            f"\n"
            f"  Steer Target: {self.servo_target_angle:+8.2f} deg\n"
            f"  Bias:     {servo_error:+8.2f} deg\n"
            f"\n"
            f"Quaternion:\n"
            f"  w: {quat[0]:+.4f}\n"
            f"  x: {quat[1]:+.4f}\n"
            f"  y: {quat[2]:+.4f}\n"
            f"  z: {quat[3]:+.4f}"
        )

        # 根据误差大小改变背景颜色
        error_color = 'lightgreen' if abs(servo_error) < 5 else ('lightyellow' if abs(servo_error) < 15 else 'lightcoral')
        self.ax_gauge.text(0.5, 0.5, angle_text, transform=self.ax_gauge.transAxes,
                          fontsize=11, va='center', ha='center', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor=error_color, alpha=0.7))

        # === 时序图 ===
        t_plot = list(self.time_history)
        r_plot = list(self.roll_history)
        p_plot = list(self.pitch_history)
        y_plot = list(self.yaw_history)

        if len(t_plot) > 1:
            for ax, data, color, title, ylim in [
                (self.ax_roll, r_plot, 'r', 'Roll', (-180, 180)),
                (self.ax_pitch, p_plot, 'g', 'Pitch', (-180, 180)),
                (self.ax_yaw, y_plot, 'b', 'Yaw', None),
            ]:
                ax.cla()
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angle (deg)')
                if ylim:
                    ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                ax.plot(t_plot, data, color + '-', linewidth=1.5)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        self.canvas.draw()
        return []

    def _draw_coordinate_frame(self, ax, R, origin=[0, 0, 0], scale=1.0, labels=['X', 'Y', 'Z']):
        """在 3D 坐标系中绘制坐标轴"""
        colors = ['red', 'green', 'blue']
        for i in range(3):
            direction = R[:, i] * scale
            ax.quiver(origin[0], origin[1], origin[2],
                     direction[0], direction[1], direction[2],
                     color=colors[i], arrow_length_ratio=0.1, linewidth=2)
            end_point = np.array(origin) + direction * 1.15
            ax.text(end_point[0], end_point[1], end_point[2], labels[i], color=colors[i], fontsize=10)

    def on_imu_select(self):
        """切换 IMU 选择"""
        self.current_imu_id = self.imu_var.get()
        # 清空历史数据
        self.time_history.clear()
        self.roll_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.servo_time_history.clear()
        self.servo_angle_history.clear()
        self.servo_imu_history.clear()
        self.start_time = time.time()

    def send_servo_angle(self):
        """发送舵机角度到串口"""
        if not self.serial_port or not self.serial_port.is_open:
            return

        try:
            angle = float(self.servo_angle_var.get())
            # 发送角度命令
            cmd = f"{int(angle)}\n"
            self.serial_port.write(cmd.encode('utf-8'))

            # 记录目标角度
            self.servo_target_angle = angle

            # 如果正在收集，记录到对比历史
            if self.imu_collector and self.imu_collector.is_collecting:
                now = time.time()
                self.servo_time_history.append(now - self.start_time)
                self.servo_angle_history.append(angle)

                # 获取当前IMU测量的角度 (使用 Roll 作为对比)
                angles, _ = self.imu_collector.get_imu_data(self.current_imu_id)
                self.servo_imu_history.append(angles[0])  # Roll

        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字角度")

    def send_quick_angle(self, angle):
        """快速发送预设角度"""
        self.servo_angle_var.set(str(angle))
        self.send_servo_angle()

    def save_data(self):
        """保存 IMU 数据"""
        if not self.imu_collector or self.imu_collector.get_count() == 0:
            messagebox.showwarning("无数据", "没有可保存的数据")
            return

        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return

        try:
            self.imu_collector.save_data(save_dir)
            messagebox.showinfo("保存成功", f"已保存 {self.imu_collector.get_count()} 帧数据到 {save_dir}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存数据时出错: {e}")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    port = config['serial']['port']
    baudrate = config['serial']['baud_rate']

    monitor = SerialMonitor(port, baudrate)
    monitor.run()