import numpy as np
import serial
import re
import time
from MPU6050.MPU6050_utils import load_config, IMUData
import math
import threading
# from imu_data_plots_module import plot_histogram

# 配置串口
config = load_config()
ser = serial.Serial(
    port=config['serial']['port'],  # 根据实际情况修改端口号
    baudrate=config['serial']['baud_rate'],  # 根据实际情况修改波特率
    timeout=config['serial']['timeout']
)


def collect_imu_raw_data(duration):
    global imu_1, imu_2, imu_3, COUNT_NUM
    start_time = time.time()
    try:
        print('imu data is collecting...')
        while time.time() - start_time < duration:
            while not (
                    imu_1.AccX != 0 and imu_1.AccY != 0 and imu_1.AccZ != 0 and imu_1.GyroX != 0 and imu_1.GyroY != 0 and imu_1.GyroZ != 0 and
                    imu_2.AccX != 0 and imu_2.AccY != 0 and imu_2.AccZ != 0 and imu_2.GyroX != 0 and imu_2.GyroY != 0 and imu_2.GyroZ != 0 and
                    imu_3.AccX != 0 and imu_3.AccY != 0 and imu_3.AccZ != 0 and imu_3.GyroX != 0 and imu_3.GyroY != 0 and imu_3.GyroZ != 0):
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        match = re.match(r"^([XYZ])(\d+):\s*Acc:\s*(-?\d+),\s*Gyro:\s*(-?\d+)$", line)
                        if match:
                            axis, imu_id, acc, gyro = match.groups()
                            # if imu_id == '3':
                            #     print(line)
                            if imu_id == '1':
                                if axis == 'X':
                                    imu_1.AccX, imu_1.GyroX = (
                                        float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[0],
                                        float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[0])
                                elif axis == 'Y':
                                    imu_1.AccY, imu_1.GyroY = (
                                        float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[1],
                                        float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[1])
                                elif axis == 'Z':
                                    imu_1.AccZ, imu_1.GyroZ = (
                                        float(acc) / imu_1.ACC_SENSITIVITY * GRAVITY - imu_1.Acc_zero_bias[2],
                                        float(gyro) / imu_1.GYRO_SENSITIVITY * DEG_TO_RAD - imu_1.Gyro_zero_bias[2])
                            elif imu_id == '2':
                                if axis == 'X':
                                    imu_2.AccX, imu_2.GyroX = (
                                        float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[0],
                                        float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[0])
                                elif axis == 'Y':
                                    imu_2.AccY, imu_2.GyroY = (
                                        float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[1],
                                        float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[1])
                                elif axis == 'Z':
                                    imu_2.AccZ, imu_2.GyroZ = (
                                        float(acc) / imu_2.ACC_SENSITIVITY * GRAVITY - imu_2.Acc_zero_bias[2],
                                        float(gyro) / imu_2.GYRO_SENSITIVITY * DEG_TO_RAD - imu_2.Gyro_zero_bias[2])

                            elif imu_id == '3':
                                if axis == 'X':
                                    imu_3.AccX, imu_3.GyroX = (
                                        float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[0],
                                        float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[0])
                                elif axis == 'Y':
                                    imu_3.AccY, imu_3.GyroY = (
                                        float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[1],
                                        float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[1])
                                elif axis == 'Z':
                                    imu_3.AccZ, imu_3.GyroZ = (
                                        float(acc) / imu_3.ACC_SENSITIVITY * GRAVITY - imu_3.Acc_zero_bias[2],
                                        float(gyro) / imu_3.GYRO_SENSITIVITY * DEG_TO_RAD - imu_3.Gyro_zero_bias[2])

            with imu_lock:
                imu_1.imu_np = np.array([imu_1.AccX, imu_1.AccY, imu_1.AccZ, imu_1.GyroX, imu_1.GyroY, imu_1.GyroZ])
                imu_2.imu_np = np.array([imu_2.AccX, imu_2.AccY, imu_2.AccZ, imu_2.GyroX, imu_2.GyroY, imu_2.GyroZ])
                imu_3.imu_np = np.array([imu_3.AccX, imu_3.AccY, imu_3.AccZ, imu_3.GyroX, imu_3.GyroY, imu_3.GyroZ])
            # print(imu_1.imu_np)

            imu_1.imu_data_df.loc[COUNT_NUM, :] = imu_1.imu_np
            imu_2.imu_data_df.loc[COUNT_NUM, :] = imu_2.imu_np
            imu_3.imu_data_df.loc[COUNT_NUM, :] = imu_3.imu_np
            imu_1.AccX, imu_1.AccY, imu_1.AccZ, imu_1.GyroX, imu_1.GyroY, imu_1.GyroZ = 0, 0, 0, 0, 0, 0
            imu_2.AccX, imu_2.AccY, imu_2.AccZ, imu_2.GyroX, imu_2.GyroY, imu_2.GyroZ = 0, 0, 0, 0, 0, 0
            imu_3.AccX, imu_3.AccY, imu_3.AccZ, imu_3.GyroX, imu_3.GyroY, imu_3.GyroZ = 0, 0, 0, 0, 0, 0
            COUNT_NUM += 1

        # return imu_data_df

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        if ser and ser.is_open:
            ser.close()
            print('serial is closed')


# initiate the IMU data
imu_1 = IMUData('mpu6050_1')
imu_2 = IMUData('mpu6050_2')
imu_3 = IMUData('mpu6050_3')

SAMPLING_FREQUENCY = config['serial']['SAMPLING_RATE']
COUNT_NUM = 0

GRAVITY = config['mpu6050_1']['Acc_params']['GRAVITY']  # m/s²

DEG_TO_RAD = math.pi / 180.0

imu_lock = threading.Lock()

if __name__ == "__main__":
    # thread 1: collect IMU data
    t1_duration = 100  # the threshold duration of Thread 1
    t1 = threading.Thread(target=collect_imu_raw_data, args=(t1_duration,), daemon=True)
    t1.start()

    MAIN_THREAD_DURATION = 5
    time.sleep(MAIN_THREAD_DURATION)

    print(imu_1.imu_data_df.shape)
    print(imu_2.imu_data_df.shape)
    print(imu_3.imu_data_df.shape)

    imu_1.save_as_csv('DataCSV/imu_1/imu_threading_data.csv')
    imu_2.save_as_csv('DataCSV/imu_2/imu_threading_data.csv')
    imu_3.save_as_csv('DataCSV/imu_3/imu_threading_data.csv')

    # plot_histogram(imu_1.imu_data_df)
