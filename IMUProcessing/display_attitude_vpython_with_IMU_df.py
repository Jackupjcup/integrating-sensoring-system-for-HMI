import numpy as np
import serial
import re
import time
from vpython import *
import pandas as pd
from MPU6050_utils import gyr_to_quat, quat_to_rot, rot_to_axes, cos_vector_angle, compute_position
import math

def estimate_traj(imu_data_df):

    Quat = np.array([1, 0, 0, 0])
    Quat_df = pd.DataFrame(columns=['w', 'x', 'y', 'z'])
    Acc_df = pd.DataFrame(columns=['Estimated_AccX', 'Estimated_AccY', 'Estimated_AccZ'])

    duration = imu_data_df.shape[0]

    for i in range(duration):
        imu_np = imu_data_df.iloc[i].values

        Quat = gyr_to_quat(Quat, imu_np[3: 6], dt=1 / SAMPLING_FREQUENCY)
        R = quat_to_rot(Quat)
        X_axis, Y_axis, Z_axis = rot_to_axes(R)

        X_acc = GRAVITY * cos_vector_angle(X_axis, GRAVITY_VEC)
        Y_acc = GRAVITY * cos_vector_angle(Y_axis, GRAVITY_VEC)
        Z_acc = GRAVITY * cos_vector_angle(Z_axis, GRAVITY_VEC)
        Acc_df.loc[i, :] = np.array([X_acc, Y_acc, Z_acc])

    return Acc_df





if __name__ == '__main__':
    imu_data_df = pd.read_csv('../DataCSV/test_imu.csv')
    length = imu_data_df.shape[0]

    SAMPLING_FREQUENCY = 100
    GRAVITY = 9.8043
    GRAVITY_VEC = np.array([0, 0, GRAVITY])

    Acc_df = estimate_traj(imu_data_df)

    # # 6 columns to analyse 'Measured Acc' and 'Estimated Acc'
    # Acc_df.loc[:, ['Measured_AccX', 'Measured_AccY', 'Measured_AccZ']] = imu_data_df.loc[:, ['AccX', 'AccY', 'AccZ']].values

    # 3 columns of bias between 'Measured Acc' and 'Estimated Acc'
    Acc_df = pd.DataFrame(np.array(Acc_df) - np.array(imu_data_df.loc[:, ['AccX', 'AccY', 'AccZ']]))


    Acc_df.to_csv('DataCSV/Accelerator_estimation.csv', index=False)
    # print(Acc_df.loc[0])


    # initiate the attitude of MPU6050
    scene = canvas(title="IMU Quaternion Attitude", width=800, height=600)
    scene.range = 2
    imu = box(length=0.5, height=1.5, width=0.2, color=color.cyan)


    Quat = np.array([1, 0, 0, 0])
    Quat_df = pd.DataFrame(columns=['w', 'x', 'y', 'z'])

    for i in range(length):
        rate(100)
        imu_np = imu_data_df.iloc[i, :].values

        Quat = gyr_to_quat(Quat, imu_np[3: 6], dt=1 / SAMPLING_FREQUENCY)
        # Quat = imu_data_df.iloc[i, :].values

        R = quat_to_rot(Quat)
        imu.axis = vector(R[0, 0], R[1, 0], R[2, 0])  # 朝向
        imu.up = vector(R[0, 1], R[1, 1], R[2, 1])  # 上方向