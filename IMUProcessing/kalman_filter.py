import numpy as np
import pandas as pd


class KalmanFilter:
    """
    单轴卡尔曼滤波器，用于融合加速度计和陀螺仪数据估计角度。

    对应 C 语言版本的 Kalman_getAngle 函数。
    状态向量: [angle, bias]
    观测量: 加速度计角度
    控制量: 陀螺仪角速度
    """

    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.Q_angle = Q_angle  # 过程噪声 - 角度
        self.Q_bias = Q_bias    # 过程噪声 - 陀螺仪零偏
        self.R_measure = R_measure  # 测量噪声 - 加速度计

        self.angle = 0.0  # 估计角度
        self.bias = 0.0   # 陀螺仪零偏估计
        self.P = np.zeros((2, 2))  # 误差协方差矩阵

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

        # 更新误差协方差矩阵 P 的预测部分
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt

        # === 更新 ===
        # 计算卡尔曼增益
        S = self.P[0][0] + self.R_measure
        K = np.array([self.P[0][0] / S, self.P[1][0] / S])

        # 利用加速度计测量值纠正预测值
        gap = new_angle - self.angle
        self.angle += K[0] * gap
        self.bias += K[1] * gap

        # 更新误差协方差矩阵
        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]
        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle


class IMUKalman:
    """
    单个 IMU 的卡尔曼姿态估计。

    融合加速度计和陀螺仪数据，输出 Roll(X), Pitch(Y), Yaw(Z) 三轴角度。
    Yaw 轴因无磁力计，使用纯陀螺仪积分，会存在漂移。

    对应 C 语言版本的 MPU6050_Read_All 中的姿态解算逻辑。
    """

    RAD_TO_DEG = 57.295779513082320876798154814105

    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.kalman_x = KalmanFilter(Q_angle, Q_bias, R_measure)  # Roll
        self.kalman_y = KalmanFilter(Q_angle, Q_bias, R_measure)  # Pitch
        self.yaw_angle = 0.0  # Yaw 纯积分

        self.KalmanAngleX = 0.0
        self.KalmanAngleY = 0.0
        self.KalmanAngleZ = 0.0

        # 原始数据 (ADC 值)
        self.Accel_X_RAW = 0
        self.Accel_Y_RAW = 0
        self.Accel_Z_RAW = 0
        self.Gyro_X_RAW = 0
        self.Gyro_Y_RAW = 0
        self.Gyro_Z_RAW = 0

        # 物理量
        self.Ax = 0.0
        self.Ay = 0.0
        self.Az = 0.0
        self.Gx = 0.0
        self.Gy = 0.0
        self.Gz = 0.0
        self.Temperature = 0.0

    def update(self, accel_raw, gyro_raw, dt):
        """
        用一组新的原始数据更新姿态估计。

        :param accel_raw: [AccX, AccY, AccZ] ADC 原始值
        :param gyro_raw:  [GyroX, GyroY, GyroZ] ADC 原始值
        :param dt: 时间步长 (s)
        """
        self.Accel_X_RAW, self.Accel_Y_RAW, self.Accel_Z_RAW = accel_raw
        self.Gyro_X_RAW, self.Gyro_Y_RAW, self.Gyro_Z_RAW = gyro_raw

        # === 从 ADC 值转为物理量 ===
        # 加速度计: ±2g → 灵敏度 16384 LSB/g
        self.Ax = self.Accel_X_RAW / 16384.0
        self.Ay = self.Accel_Y_RAW / 16384.0
        self.Az = self.Accel_Z_RAW / 14418.0  # Z轴校正值（与 C 代码一致）

        # 陀螺仪: ±250°/s → 灵敏度 131 LSB/(°/s)
        self.Gx = self.Gyro_X_RAW / 131.0
        self.Gy = self.Gyro_Y_RAW / 131.0
        self.Gz = self.Gyro_Z_RAW / 131.0 - 4  # Z轴零偏补偿（与 C 代码一致）

        # 死区处理
        if -1 < self.Gz < 1:
            self.Gz = 0.0

        # === Roll (X轴) ===
        roll_sqrt = np.sqrt(self.Accel_X_RAW ** 2 + self.Accel_Z_RAW ** 2)
        if roll_sqrt != 0.0:
            roll = np.arctan(self.Accel_Y_RAW / roll_sqrt) * self.RAD_TO_DEG
        else:
            roll = 0.0

        # === Pitch (Y轴) ===
        pitch = np.arctan2(-self.Accel_X_RAW, self.Accel_Z_RAW) * self.RAD_TO_DEG

        # Pitch 跨 ±90° 时重置卡尔曼状态（防止角度跳变）
        if (pitch < -90 and self.KalmanAngleY > 90) or (pitch > 90 and self.KalmanAngleY < -90):
            self.kalman_y.angle = pitch
            self.KalmanAngleY = pitch
        else:
            self.KalmanAngleY = self.kalman_y.get_angle(pitch, self.Gy, dt)

        # Pitch 超过 ±90° 时翻转 Roll 的陀螺仪符号
        if abs(self.KalmanAngleY) > 90:
            self.Gx = -self.Gx

        self.KalmanAngleX = self.kalman_x.get_angle(roll, self.Gx, dt)

        # === Yaw (Z轴) - 纯陀螺仪积分 ===
        self.yaw_angle += self.Gz * dt
        self.KalmanAngleZ = self.yaw_angle

    def get_angles(self):
        """返回 [Roll, Pitch, Yaw] 角度值 (deg)"""
        return np.array([self.KalmanAngleX, self.KalmanAngleY, self.KalmanAngleZ])


def parse_serial_line(line):
    """
    解析串口数据行，如 "X1A-124G38"

    :param line: 串口数据字符串
    :return: (axis, imu_id, accel, gyro) 或 None
    """
    line = line.strip()
    if not line:
        return None

    try:
        axis = line[0]           # X, Y, Z
        imu_id = int(line[1])    # 1, 2, 3
        a_idx = line.index('A')
        g_idx = line.index('G')
        accel = int(line[a_idx + 1:g_idx])
        gyro = int(line[g_idx + 1:])
        return (axis, imu_id, accel, gyro)
    except (ValueError, IndexError):
        return None


def parse_imu_frame(lines):
    """
    解析一个完整的 IMU 帧（6行：X1,Y1,Z1,X2,Y2,Z2）。

    :param lines: 6行串口数据
    :return: dict, 3个IMU各自的 [AccX, AccY, AccZ] 和 [GyroX, GyroY, GyroZ]
    """
    imu_data = {}
    for line in lines:
        parsed = parse_serial_line(line)
        if parsed is None:
            continue
        axis, imu_id, accel, gyro = parsed
        if imu_id not in imu_data:
            imu_data[imu_id] = {'accel': [0, 0, 0], 'gyro': [0, 0, 0]}
        idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        imu_data[imu_id]['accel'][idx] = accel
        imu_data[imu_id]['gyro'][idx] = gyro

    return imu_data


class IMUManager:
    """
    管理 3 个 IMU 传感器，使用卡尔曼滤波进行姿态估计。
    数据存储使用 numpy 和 pandas。
    """

    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.imus = {
            1: IMUKalman(Q_angle, Q_bias, R_measure),
            2: IMUKalman(Q_angle, Q_bias, R_measure),
            3: IMUKalman(Q_angle, Q_bias, R_measure),
        }

        # pandas 存储原始数据
        self.raw_df = pd.DataFrame(
            columns=['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ', 'IMU1_GyroX', 'IMU1_GyroY', 'IMU1_GyroZ',
                      'IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ', 'IMU2_GyroX', 'IMU2_GyroY', 'IMU2_GyroZ',
                      'IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ', 'IMU3_GyroX', 'IMU3_GyroY', 'IMU3_GyroZ']
        )

        # pandas 存储滤波后角度
        self.angle_df = pd.DataFrame(
            columns=['IMU1_Roll', 'IMU1_Pitch', 'IMU1_Yaw',
                      'IMU2_Roll', 'IMU2_Pitch', 'IMU2_Yaw',
                      'IMU3_Roll', 'IMU3_Pitch', 'IMU3_Yaw']
        )

        # numpy 存储原始数据 (N×18)
        self.raw_np = np.zeros((0, 18))

        # numpy 存储滤波后角度 (N×9)
        self.angle_np = np.zeros((0, 9))

        # 记录上一次时间戳，用于计算 dt
        self._last_time = None

    def update(self, imu_data_dict, timestamp=None):
        """
        更新所有 IMU 的姿态估计并存储数据。

        :param imu_data_dict: parse_imu_frame 的返回值
            {imu_id: {'accel': [Ax, Ay, Az], 'gyro': [Gx, Gy, Gz]}}
        :param timestamp: 当前时间戳 (秒)，用于计算 dt
        """
        # 计算 dt
        if timestamp is not None and self._last_time is not None:
            dt = timestamp - self._last_time
            if dt <= 0:
                dt = 0.005  # 默认 5ms
        else:
            dt = 0.005  # 默认 5ms (200Hz)
        self._last_time = timestamp

        raw_row = []
        angle_row = []

        for imu_id in [1, 2, 3]:
            if imu_id not in imu_data_dict:
                raw_row.extend([0, 0, 0, 0, 0, 0])
                angle_row.extend([0.0, 0.0, 0.0])
                continue

            accel = imu_data_dict[imu_id]['accel']
            gyro = imu_data_dict[imu_id]['gyro']

            # 卡尔曼滤波更新
            self.imus[imu_id].update(accel, gyro, dt)
            angles = self.imus[imu_id].get_angles()

            raw_row.extend([accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]])
            angle_row.extend([angles[0], angles[1], angles[2]])

        # 存入 pandas
        self.raw_df.loc[len(self.raw_df)] = raw_row
        self.angle_df.loc[len(self.angle_df)] = angle_row

        # 存入 numpy
        self.raw_np = np.vstack([self.raw_np, raw_row]) if self.raw_np.size else np.array([raw_row])
        self.angle_np = np.vstack([self.angle_np, angle_row]) if self.angle_np.size else np.array([angle_row])

    def reset(self):
        """清空所有存储数据，重置滤波器状态。"""
        for imu in self.imus.values():
            imu.kalman_x = KalmanFilter()
            imu.kalman_y = KalmanFilter()
            imu.yaw_angle = 0.0
            imu.KalmanAngleX = 0.0
            imu.KalmanAngleY = 0.0
            imu.KalmanAngleZ = 0.0

        self.raw_df = pd.DataFrame(columns=self.raw_df.columns)
        self.angle_df = pd.DataFrame(columns=self.angle_df.columns)
        self.raw_np = np.zeros((0, 18))
        self.angle_np = np.zeros((0, 9))
        self._last_time = None


# ===== 测试 =====
if __name__ == '__main__':
    # 模拟串口数据
    test_lines = [
        "X1A-124G38",
        "Y1A2056G49",
        "Z1A-137G-16",
        "X2A480G-57",
        "Y2A-128G-34",
        "Z2A1626G2",
        "X3A336G-647",
        "Y3A-198G-43",
        "Z3A1904G-26",
    ]

    # 解析数据
    imu_data = parse_imu_frame(test_lines)

    # 创建 IMU 管理器
    manager = IMUManager()

    # 更新姿态估计
    manager.update(imu_data)

    # 输出结果
    print("=== 原始数据 (numpy) ===")
    print(manager.raw_np)

    print("\n=== 滤波后角度 (numpy) ===")
    print(manager.angle_np)

    print("\n=== 原始数据 (DataFrame) ===")
    print(manager.raw_df)

    print("\n=== 滤波后角度 (DataFrame) ===")
    print(manager.angle_df)

    # 模拟连续更新（200Hz，1秒）
    import time
    print("\n=== 连续模拟 100 步 ===")
    t_start = time.time()
    for i in range(100):
        t = t_start + i * 0.005
        manager.update(imu_data, timestamp=t)

    print(f"数据点数: {len(manager.angle_df)}")
    print(f"numpy shape: raw={manager.raw_np.shape}, angle={manager.angle_np.shape}")
    print("\n最后5帧角度:")
    print(manager.angle_df.tail())
