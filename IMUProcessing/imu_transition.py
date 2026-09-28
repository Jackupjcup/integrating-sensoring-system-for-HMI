import matplotlib.pyplot as plt
import numpy as np
import serial
import re
import time
from vpython import *  
import pandas as pd
from MPU6050_utils import load_config, quat_to_rot, IMUData
import math

# cd D:\pycharm_projects\SmartGlove\MPU6050
# python imu_transition.py


config = load_config("config.json")
# 配置串口
ser = serial.Serial(
    port=config['serial']['port'],  # 根据实际情况修改端口号
    baudrate=config['serial']['baud_rate'],  # 根据实际情况修改波特率
    timeout=config['serial']['timeout']
)

def process_data(line, count):
    global imu_1, imu_2, imu_3
    match = re.match(r'^([XYZ])(\d)A(-?\d+)G(-?\d+)$', line)
    if match:
        axis, imu_id, acc, gyro = match.groups()
        # print(match.groups())
        if imu_id == '1':
            if axis == 'X':
                imu_1.AccX, imu_1.GyroX = (float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[0],
                                           float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[0])
            elif axis == 'Y':
                imu_1.AccY, imu_1.GyroY = (float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[1],
                                           float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[1])
            elif axis == 'Z':
                imu_1.AccZ, imu_1.GyroZ = (float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[2],
                                           float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[2])
        elif imu_id == '2':
            if axis == 'X':
                imu_2.AccX, imu_2.GyroX = (float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[0],
                                           float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[0])
            elif axis == 'Y':
                imu_2.AccY, imu_2.GyroY = (float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[1],
                                           float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[1])
            elif axis == 'Z':
                imu_2.AccZ, imu_2.GyroZ = (float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[2],
                                           float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[2])

        elif imu_id == '3':
            if axis == 'X':
                imu_3.AccX, imu_3.GyroX = (float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[0],
                                           float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[0])
            elif axis == 'Y':
                imu_3.AccY, imu_3.GyroY = (float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[1],
                                           float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[1])
            elif axis == 'Z':
                imu_3.AccZ, imu_3.GyroZ = (float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[2],
                                           float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[2])

    if (imu_1.AccX != 0 and imu_1.AccY != 0 and imu_1.AccZ != 0 and imu_1.GyroX != 0 and imu_1.GyroY != 0 and imu_1.GyroZ != 0 and
        imu_2.AccX != 0 and imu_2.AccY != 0 and imu_2.AccZ != 0 and imu_2.GyroX != 0 and imu_2.GyroY != 0 and imu_2.GyroZ != 0 and
        imu_3.AccX != 0 and imu_3.AccY != 0 and imu_3.AccZ != 0 and imu_3.GyroX != 0 and imu_3.GyroY != 0 and imu_3.GyroZ != 0):

        imu_1.imu_np = np.array([imu_1.AccX, imu_1.AccY, imu_1.AccZ, imu_1.GyroX, imu_1.GyroY, imu_1.GyroZ])
        imu_2.imu_np = np.array([imu_2.AccX, imu_2.AccY, imu_2.AccZ, imu_2.GyroX, imu_2.GyroY, imu_2.GyroZ])
        imu_3.imu_np = np.array([imu_3.AccX, imu_3.AccY, imu_3.AccZ, imu_3.GyroX, imu_3.GyroY, imu_3.GyroZ])
        # print(f'{imu_3.GyroX}, {imu_3.GyroY}, {imu_3.GyroZ}')

        imu_1.imu_data_df.loc[count, :] = imu_1.imu_np
        imu_2.imu_data_df.loc[count, :] = imu_2.imu_np
        imu_3.imu_data_df.loc[count, :] = imu_3.imu_np

        imu_1.AccX, imu_1.AccY, imu_1.AccZ, imu_1.GyroX, imu_1.GyroY, imu_1.GyroZ = 0, 0, 0, 0, 0, 0
        imu_2.AccX, imu_2.AccY, imu_2.AccZ, imu_2.GyroX, imu_2.GyroY, imu_2.GyroZ = 0, 0, 0, 0, 0, 0
        imu_3.AccX, imu_3.AccY, imu_3.AccZ, imu_3.GyroX, imu_3.GyroY, imu_3.GyroZ = 0, 0, 0, 0, 0, 0
        count += 1
    else:
        imu_1.imu_np = np.array([imu_1.AccX, imu_1.AccY, imu_1.AccZ, imu_1.GyroX, imu_1.GyroY, imu_1.GyroZ])
        imu_2.imu_np = np.array([imu_2.AccX, imu_2.AccY, imu_2.AccZ, imu_2.GyroX, imu_2.GyroY, imu_2.GyroZ])
        imu_3.imu_np = np.array([imu_3.AccX, imu_3.AccY, imu_3.AccZ, imu_3.GyroX, imu_3.GyroY, imu_3.GyroZ])
    return count

if __name__ == "__main__":
    # initiate the IMU data
    imu_1 = IMUData('mpu6050_1', "config.json")
    imu_2 = IMUData('mpu6050_2', "config.json")
    imu_3 = IMUData('mpu6050_3', "config.json")

    SAMPLING_FREQUENCY = config['serial']['SAMPLING_RATE']

    GRAVITY = config['mpu6050_1']['Acc_params']['GRAVITY']  # m/s²
    DEG_TO_RAD = math.pi / 180.0

    COUNT_NUM = 0
    dt_list = []

    # initiate the attitude of MPU6050
    scene = canvas(title="IMU Quaternion Attitude", width=800, height=600)
    scene.range = 2
    imu = box(length=0.5, height=1.5, width=0.2, color=color.cyan)

    # initiate position and velocity
    Pos = np.zeros(3)
    Vel = np.zeros(3)

    # # initiate ahrs  filters
    # madgwick = ahrs.filters.Madgwick(frequency=config['serial']['SAMPLING_RATE'])

    # initial Calibration of the Gyro

    START_TIME = time.time()
    animation_duration = 30

    try:
        print('imu data is collecting...')
        last_time = time.time()
        while time.time() - START_TIME < animation_duration:
            rate(1000)
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    # print(line)
                    last_num = COUNT_NUM
                    COUNT_NUM = process_data(line, COUNT_NUM)
                    if last_num != COUNT_NUM:
                        now = time.time()
                        dt = now - last_time
                        dt_list.append(dt)
                        last_time = now
                        imu_1.gyro_to_quat(dt=dt)
                        imu_2.gyro_to_quat(dt=dt)
                        imu_3.gyro_to_quat(dt=dt)
                        imu_1.Quat_df.loc[COUNT_NUM, :] = imu_1.Quat
                        imu_2.Quat_df.loc[COUNT_NUM, :] = imu_2.Quat
                        imu_3.Quat_df.loc[COUNT_NUM, :] = imu_3.Quat
                        # Quat = madgwick.updateIMU(q=Quat, gyr=imu_np[3: 6], acc=imu_np[0: 3])
                        # Quat_df.loc[COUNT_NUM] = Quat
                        # Quat = ekf.step(imu_np[: 3], imu_np[3:])

                        R = quat_to_rot(imu_3.Quat)
                        imu.axis = vector(R[0,0], R[1,0], R[2,0])   # 朝向
                        imu.up = vector(R[0,1], R[1,1], R[2,1])   # 上方向

                        # Vel, Pos = compute_position(imu_np[0: 3], Quat,
                        #                             dt=1/SAMPLING_FREQUENCY,
                        #                             prev_vel=Vel,
                        #                             prev_pos=Pos)
                        # imu.pos = vector(Pos[0], Pos[1], Pos[2])
    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        ser.close()
        print('serial is closed')
        print(COUNT_NUM)
        print(f'mean of sampling time is {np.mean(dt_list)}')

        imu_1.save_as_csv('DataCSV/imu_1/imu_animation_data.csv')
        imu_2.save_as_csv('DataCSV/imu_2/imu_animation_data.csv')
        imu_3.save_as_csv('DataCSV/imu_3/imu_animation_data.csv')

        imu_1.Quat_df.to_csv("DataCSV/imu_1/imu_animation_Quat.csv", index=False)
        imu_2.Quat_df.to_csv("DataCSV/imu_2/imu_animation_Quat.csv", index=False)
        imu_3.Quat_df.to_csv("DataCSV/imu_3/imu_animation_Quat.csv", index=False)




