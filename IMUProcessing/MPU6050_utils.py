import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
import os
import numpy as np


# 读取 self.config.json 文件
def load_config(config_path_str):
    # 获取配置文件的绝对路径（避免路径问题）
    # config_path = os.path.join(os.path.dirname(__file__), config_path_str)

    # Simplified Version but More Flexile
    config_path = config_path_str
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # 解析 JSON 内容为字典
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"错误：{config_path_str} 文件不存在")
        return None
    except json.JSONDecodeError:
        print(f"错误：{config_path_str} 语法格式不正确")
        return None


class KalmanFilter:
    """
    单轴卡尔曼滤波器，用于融合加速度计和陀螺仪数据估计角度。
    对应 C 语言版本的 Kalman_getAngle 函数。
    """
    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.Q_angle = Q_angle      # 过程噪声 - 角度
        self.Q_bias = Q_bias        # 过程噪声 - 陀螺仪零偏
        self.R_measure = R_measure  # 测量噪声 - 加速度计
        self.angle = 0.0            # 估计角度
        self.bias = 0.0             # 陀螺仪零偏估计
        self.P = np.zeros((2, 2))   # 误差协方差矩阵

    def get_angle(self, new_angle, new_rate, dt):
        """
        一步卡尔曼滤波更新。
        :param new_angle: 加速度计计算的角度 (deg)
        :param new_rate: 陀螺仪角速度 (deg/s)
        :param dt: 时间步长 (s)
        :return: 融合后的角度 (deg)
        """
        # === 预测 ===
        rate = new_rate - self.bias
        self.angle += dt * rate

        # 更新误差协方差矩阵
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt

        # === 更新 ===
        S = self.P[0][0] + self.R_measure
        K = np.array([self.P[0][0] / S, self.P[1][0] / S])

        gap = new_angle - self.angle
        self.angle += K[0] * gap
        self.bias += K[1] * gap

        P00_temp, P01_temp = self.P[0][0], self.P[0][1]
        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle


class IMUData:
    def __init__(self, imu_id_str, config_path):
        self.imu_id_str = imu_id_str
        self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ = 0, 0, 0, 0, 0, 0
        self.IMU_data_dic = self.init_imu_data_dic()
        self.Quat = np.array([1, 0, 0, 0])
        self.Quat_df = pd.DataFrame(columns=['w', 'x', 'y', 'z'])
        self.mean_AccX, self.mean_AccY, self.mean_AccZ, self.mean_GyroX, self.mean_GyroY, self.mean_GyroZ = 0, 0, 0, 0, 0, 0
        self.config = load_config(config_path)

        # Gyro stationary bias
        self.GyroX_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroX_zero_bias']
        self.GyroY_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroY_zero_bias']
        self.GyroZ_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroZ_zero_bias']
        self.Gyro_zero_bias = np.array([self.GyroX_zero_bias, self.GyroY_zero_bias, self.GyroZ_zero_bias])

        # Acc stationary(Zero) bias
        self.AccX_zero_bias = self.config[self.imu_id_str]['Acc_params']['AccX_zero_bias']
        self.AccY_zero_bias = self.config[self.imu_id_str]['Acc_params']['AccY_zero_bias']
        self.AccZ_zero_bias = self.config[self.imu_id_str]['Acc_params']['AccZ_zero_bias']
        self.Acc_zero_bias = np.array([self.AccX_zero_bias, self.AccY_zero_bias, self.AccZ_zero_bias])

        # 加速度计：±16g → 灵敏度 2048 LSB/g；重力加速度 g = 9.80665 m/s²
        self.ACC_SENSITIVITY = self.config[self.imu_id_str]['Acc_params']['ACC_SENSITIVITY']

        # 陀螺仪：±2000 °/s → 灵敏度 16.4 LSB/(°/s)
        self.GYRO_SENSITIVITY = self.config[self.imu_id_str]['Gyro_params']['GYRO_SENSITIVITY']

        # Raw Data
        self.imu_data_df = pd.DataFrame(columns=['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
        self.imu_np = np.array([self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ])

        # Euler Degree of X, Y, Z
        self.X_Euler = 0.0
        self.Y_Euler = 0.0
        self.Z_Euler = 0.0
        self.Euler_Degree = pd.DataFrame(columns=['Roll', 'Pitch', 'Yaw'])

        # ===== 卡尔曼滤波器 =====
        self.kalman_roll = KalmanFilter(Q_angle=0.001, Q_bias=0.003, R_measure=0.03)
        self.kalman_pitch = KalmanFilter(Q_angle=0.001, Q_bias=0.003, R_measure=0.03)
        self.yaw_angle = 0.0  # Yaw 纯陀螺仪积分

        # 卡尔曼滤波后的欧拉角 (deg)
        self.KalmanRoll = 0.0
        self.KalmanPitch = 0.0
        self.KalmanYaw = 0.0

        # 存储 Kalman 角度的 DataFrame
        self.KalmanAngle_df = pd.DataFrame(columns=['Roll', 'Pitch', 'Yaw'])

        # 存储 Kalman 角度的 numpy 数组 (N×3)
        self.KalmanAngle_np = np.zeros((0, 3))

    def init_imu_data_dic(self):
        return {"AccX": self.AccX, "AccY": self.AccY, "AccZ": self.AccZ, "GyroX": self.GyroX, "GyroY": self.GyroY, "GyroZ": self.GyroZ}
    def collect_imu_data(self):
        self.AccX = self.IMU_data_dic['AccX']
        self.AccY = self.IMU_data_dic['AccY']
        self.AccZ = self.IMU_data_dic['AccZ']
        self.GyroX = self.IMU_data_dic['GyroX']
        self.GyroY = self.IMU_data_dic['GyroY']
        self.GyroZ = self.IMU_data_dic['GyroZ']
        self.imu_np = np.array([self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ])


    def clear_dic(self):
        self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ = 0, 0, 0, 0, 0, 0
        self.IMU_data_dic = self.init_imu_data_dic()

    def reset(self):
        self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ = 0, 0, 0, 0, 0, 0
        self.IMU_data_dic = self.init_imu_data_dic()
        self.mean_AccX, self.mean_AccY, self.mean_AccZ, self.mean_GyroX, self.mean_GyroY, self.mean_GyroZ = 0, 0, 0, 0, 0, 0
        self.Quat = np.array([1, 0, 0, 0])
        self.Quat_df = pd.DataFrame(columns=['w', 'x', 'y', 'z'])

        # Gyro stationary bias
        self.GyroX_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroX_zero_bias']
        self.GyroY_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroY_zero_bias']
        self.GyroZ_zero_bias = self.config[self.imu_id_str]['Gyro_params']['GyroZ_zero_bias']

        self.imu_data_df = pd.DataFrame(columns=['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
        self.imu_np = np.array([self.AccX, self.AccY, self.AccZ, self.GyroX, self.GyroY, self.GyroZ])

        # 重置卡尔曼滤波器
        self.kalman_roll = KalmanFilter(Q_angle=0.001, Q_bias=0.003, R_measure=0.03)
        self.kalman_pitch = KalmanFilter(Q_angle=0.001, Q_bias=0.003, R_measure=0.03)
        self.yaw_angle = 0.0
        self.KalmanRoll, self.KalmanPitch, self.KalmanYaw = 0.0, 0.0, 0.0
        self.KalmanAngle_df = pd.DataFrame(columns=['Roll', 'Pitch', 'Yaw'])
        self.KalmanAngle_np = np.zeros((0, 3))

    def calculate_mean_of_data(self):
        self.mean_AccX = np.mean(self.imu_data_df.iloc[:, 0])
        self.mean_AccY = np.mean(self.imu_data_df.iloc[:, 1])
        self.mean_AccZ = np.mean(self.imu_data_df.iloc[:, 2])

        self.mean_GyroX = np.mean(self.imu_data_df.iloc[:, 3])
        self.mean_GyroY = np.mean(self.imu_data_df.iloc[:, 4])
        self.mean_GyroZ = np.mean(self.imu_data_df.iloc[:, 5])

    def gyro_to_quat(self, gyro=None, dt=1/200):
        if gyro is not None and len(gyro) == 3:
            self.Quat = euler_to_quat(*gyro)
        else:
            gyr_np = self.imu_np[3: 6]

            quat = self.Quat

            gx, gy, gz = (self.config[self.imu_id_str]['Gyro_params']['GyroX_axis_coef'] * gyr_np[self.config[self.imu_id_str]['Gyro_params']['GyroX_axis']],
                          self.config[self.imu_id_str]['Gyro_params']['GyroY_axis_coef'] * gyr_np[self.config[self.imu_id_str]['Gyro_params']['GyroY_axis']],
                          self.config[self.imu_id_str]['Gyro_params']['GyroZ_axis_coef'] * gyr_np[self.config[self.imu_id_str]['Gyro_params']['GyroZ_axis']])

            # 陀螺仪角速度转四元数微分 dq = 0.5 * q ⊗ [0, wx, wy, wz]
            omega = np.array([0, gx, gy, gz])
            dq = 0.5 * quat_mult(quat, omega)

            # 积分 q = q + dq*dt

            quat = quat + dq * dt
            quat_ = normalize_quat(quat)

            self.Quat = quat_


    def kalman_update_angles(self, dt=1/200):
        """
        使用卡尔曼滤波融合加速度计和陀螺仪数据，计算 Roll/Pitch/Yaw 角度。
        对应 C 语言版本的 MPU6050_Read_All 中的姿态解算逻辑。

        :param dt: 时间步长 (s)，默认 1/200 = 0.005s
        """
        RAD_TO_DEG = 57.295779513082320876798154814105

        # 从已存储的数据获取加速度和陀螺仪值 (m/s² 和 rad/s)
        ax, ay, az = self.AccX, self.AccY, self.AccZ
        gx, gy, gz = self.GyroX, self.GyroY, self.GyroZ

        # 陀螺仪 rad/s → deg/s
        gx_deg = np.rad2deg(gx)
        gy_deg = np.rad2deg(gy)
        gz_deg = np.rad2deg(gz)

        # 添加死区
        GYRO_DEADZONE = 1.0  # deg/s
        if abs(gz_deg) < GYRO_DEADZONE:
            gz_deg = 0.0

        # 加速度 m/s² → g (归一化用于角度计算)
        ax_g = ax / 9.8043
        ay_g = ay / 9.8043
        az_g = az / 9.8043

        # === Roll (X轴) ===
        roll_sqrt = np.sqrt(ax_g ** 2 + az_g ** 2)
        if roll_sqrt > 1e-6:
            roll_acc = np.arctan2(ay_g, roll_sqrt) * RAD_TO_DEG
        else:
            roll_acc = 0.0

        # === Pitch (Y轴) ===
        pitch_acc = np.arctan2(-ax_g, az_g) * RAD_TO_DEG

        # Pitch 跨 ±90° 时重置卡尔曼状态
        if (pitch_acc < -90 and self.KalmanPitch > 90) or (pitch_acc > 90 and self.KalmanPitch < -90):
            self.kalman_pitch.angle = pitch_acc
            self.KalmanPitch = pitch_acc
        else:
            self.KalmanPitch = self.kalman_pitch.get_angle(pitch_acc, gy_deg, dt)

        # Pitch 超过 ±90° 时翻转 Roll 的陀螺仪符号
        if abs(self.KalmanPitch) > 90:
            gx_deg = -gx_deg

        self.KalmanRoll = self.kalman_roll.get_angle(roll_acc, gx_deg, dt)

        # === Yaw (Z轴) - 纯陀螺仪积分 ===
        self.yaw_angle += gz_deg * dt
        self.KalmanYaw = self.yaw_angle

        # Output X, Y axis Raw Degree
        self.X_Euler += gx_deg * dt
        self.Y_Euler += gy_deg * dt
        self.Z_Euler += gz_deg * dt
        self.Euler_Degree.loc[len(self.Euler_Degree)] = [self.X_Euler, self.Y_Euler, self.Z_Euler]

        # 存储到 DataFrame 和 numpy
        self.KalmanAngle_df.loc[len(self.KalmanAngle_df)] = [self.KalmanRoll, self.KalmanPitch, self.KalmanYaw]
        new_row = np.array([[self.KalmanRoll, self.KalmanPitch, self.KalmanYaw]])
        self.KalmanAngle_np = np.vstack([self.KalmanAngle_np, new_row]) if self.KalmanAngle_np.size else new_row

    def get_kalman_angles(self):
        """返回卡尔曼滤波后的 [Roll, Pitch, Yaw] 角度 (deg)"""
        return np.array([self.KalmanRoll, self.KalmanPitch, self.KalmanYaw])

    def get_raw_euler_degrees(self):
        return np.array([self.X_Euler, self.Y_Euler, self.Z_Euler])

    def save_as_csv(self, file_path, whether_print_result=False):
        if not file_path:
            print("错误：文件路径不能为空！")
            return False

        try:
            # 提取目录路径
            dir_path = os.path.dirname(file_path)
            # 递归创建目录（若不存在）
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            # 保存CSV（指定编码避免中文乱码）
            self.imu_data_df.to_csv(file_path, index=False, encoding='utf-8')
            if whether_print_result:
                print(f"{self.imu_id_str} has been saved at {os.path.abspath(file_path)}")
            return True

        except Exception as e:
            print(f"保存CSV文件失败：{str(e)}")
            return False


def normalize_quat(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.zeros_like(q)

def quat_mult(q1, q2):
    """四元数乘法 q = q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def gyr_to_quat(quat, gyr_np, dt=0.002):
    """
    使用陀螺仪积分得到姿态四元数
    gyro_data: Nx3, rad/s
    dt: 时间步长
    """
    # quat = np.array([1, 0, 0, 0], dtype=float)  # 初始四元数
    # quat_list = []
    gx, gy, gz = gyr_np[0], gyr_np[1], gyr_np[2]

    # 陀螺仪角速度转四元数微分 dq = 0.5 * q ⊗ [0, wx, wy, wz]
    omega = np.array([0, gx, gy, gz])
    dq = 0.5 * quat_mult(quat, omega)

    # 积分 q = q + dq*dt

    quat = quat + dq * dt
    quat_ = normalize_quat(quat)

    return quat_

def quat_to_rot(q):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])
    return R


def rot_to_axes(R):
    """
    从旋转矩阵提取物体XYZ轴在世界坐标系中的指向
    :param R: 3×3旋转矩阵（quat_to_rot输出）
    :return: x_axis, y_axis, z_axis（均为单位向量，shape=(3,)）
    """
    # 取旋转矩阵的列向量作为轴方向
    x_axis = R[:, 0]  # 物体X轴指向
    y_axis = R[:, 1]  # 物体Y轴指向
    z_axis = R[:, 2]  # 物体Z轴指向

    # 归一化（避免浮点误差导致非单位向量）
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    return x_axis, y_axis, z_axis


import numpy as np


def cos_vector_angle(a, b, return_deg=True):
    """
    计算两个三维向量的夹角
    :param a: 向量a，numpy数组/列表，shape=(3,)
    :param b: 向量b，numpy数组/列表，shape=(3,)
    :param return_deg: 是否返回角度（True=角度，False=弧度）
    :return: 两向量的最小夹角（范围0~180°或0~π）
    """

    # 步骤1：计算模长（避免除以0）
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        raise ValueError("输入向量不能为零向量（模长接近0）")

    # 步骤2：计算点积
    dot_product = np.dot(a, b)

    # 步骤3：计算cosθ（裁剪到[-1,1]，避免浮点误差导致arccos报错）
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 关键：防止超出arccos定义域

    # # 步骤4：计算夹角
    # theta_rad = np.arccos(cos_theta)
    #
    # # 转换为角度（若需要）
    # if return_deg:
    #     theta_deg = np.rad2deg(theta_rad)
    #     return theta_deg
    # else:
    #     return theta_rad
    return cos_theta


def compute_position(acc_body, quat, dt, prev_vel, prev_pos):
    # 四元数转换为旋转矩阵
    R_wb = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    # 将加速度从机体系转到世界坐标系
    acc_world = R_wb @ acc_body

    # 重力补偿
    g = np.array([0, 0, 9.8043])
    acc_lin = acc_world - g

    # ---- 两次积分 ----
    vel = prev_vel + acc_lin * dt
    pos = prev_pos + vel * dt

    return vel, pos


def euler_to_quat(roll, pitch, yaw):
    """欧拉角 (deg) → 四元数 [w, x, y, z]"""
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    q = rot.as_quat()  # 返回 [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # 转为 [w, x, y, z]