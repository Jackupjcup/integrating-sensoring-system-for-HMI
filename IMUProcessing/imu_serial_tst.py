import numpy as np
import serial
import re
import time
import pandas as pd
import math
from plots_module import plot_histogram

# cd C:\Users\Admin\PycharmProjects\smart glove\MPU6050
# python imu_transition.py

# 配置串口
ser = serial.Serial(
    port='COM7',  # 根据实际情况修改端口号
    baudrate=460800,  # 根据实际情况修改波特率
    timeout=1
)







if __name__ == "__main__":
    # 初始化变量
    AccX, AccY, AccZ, GyroX, GyroY, GyroZ = 0, 0, 0, 0, 0, 0
    Quat = np.array([1, 0, 0, 0])
    Quat_df = pd.DataFrame(columns=['w', 'x', 'y', 'z'])
    SAMPLING_FREQUENCY = 100

    # initiate position and velocity
    Pos = np.zeros(3)
    Vel = np.zeros(3)

    # Gyro stationary bias
    GyroX_zero_bias = 0.0413
    GyroY_zero_bias = 0.0538
    GyroZ_zero_bias = -0.0192
    Gyro_zero_bias = np.array([GyroX_zero_bias, GyroY_zero_bias, GyroZ_zero_bias])

    # 加速度计：±16g → 灵敏度 2048 LSB/g；重力加速度 g = 9.80665 m/s²
    ACC_SENSITIVITY = 2048.0  # LSB/g
    GRAVITY = 9.80665  # m/s²

    # 陀螺仪：±2000 °/s → 灵敏度 16.4 LSB/(°/s)；角度转换系数：π/180（°→rad）
    GYRO_SENSITIVITY = 16.4  # LSB/(°/s)
    DEG_TO_RAD = math.pi / 180.0

    COUNT_NUM = 0
    Count = 0
    imu_data_df = pd.DataFrame(columns=['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
    START_TIME = time.time()
    animation_duration = 5

    # test_on_dataset parameters
    time_list = []
    not_sequence_count = 0

    try:
        print('imu data is collecting...')
        while time.time() - START_TIME < animation_duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                Count += 1
                if line:
                    print(line)
                else:
                    print("line is none")
    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        ser.close()
        print('serial is closed')

        # imu_data_df.to_csv('DataCSV/imu_raw_data.csv', index=False)
        # print('imu_transition.csv has been saved')
        #
        # print(imu_data_df.shape[0])
        # print(f'the num of not sequence is {not_sequence_count}')









