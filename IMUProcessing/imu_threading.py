import json

import numpy as np
import serial
import re
import time
from MPU6050_utils import load_config, IMUData
import math
import threading

#
config = load_config("config.json")



class ThreeIMUCollector:
    def __init__(self, ser, imu_1, imu_2, imu_3, count_num, duration):
        self.imu_dic = {'imu_1': imu_1,
                        'imu_2': imu_2,
                        'imu_3': imu_3}
        self.COUNT_NUM = count_num
        self.duration = duration
        self.serial = ser

        self.t1 = None

        self.is_collecting = False
        self.main_thread_flag = False
        self.stop_event = threading.Event()

    def collect_imu_raw_data(self):
        # self.is_collecting = True
        try:
            print('imu data is collecting...')
            last_time = time.time()
            # while (time.time() - start_time < self.duration) and self.is_collecting:
            while not self.stop_event.is_set():
                while not (
                        self.imu_dic['imu_1'].AccX != 0 and
                        self.imu_dic['imu_1'].AccY != 0 and
                        self.imu_dic['imu_1'].AccZ != 0 and
                        self.imu_dic['imu_1'].GyroX != 0 and
                        self.imu_dic['imu_1'].GyroY != 0 and
                        self.imu_dic['imu_1'].GyroZ != 0 and

                        self.imu_dic['imu_2'].AccX != 0 and
                        self.imu_dic['imu_2'].AccY != 0 and
                        self.imu_dic['imu_2'].AccZ != 0 and
                        self.imu_dic['imu_2'].GyroX != 0 and
                        self.imu_dic['imu_2'].GyroY != 0 and
                        self.imu_dic['imu_2'].GyroZ != 0 and

                        self.imu_dic['imu_3'].AccX != 0 and
                        self.imu_dic['imu_3'].AccY != 0 and
                        self.imu_dic['imu_3'].AccZ != 0 and
                        self.imu_dic['imu_3'].GyroX != 0 and
                        self.imu_dic['imu_3'].GyroY != 0 and
                        self.imu_dic['imu_3'].GyroZ != 0):
                    if self.serial.in_waiting > 0:
                        line = self.serial.readline().decode('utf-8').strip()
                        if line:
                            match = re.match(r"^([XYZ])(\d+):\s*Acc:\s*(-?\d+),\s*Gyro:\s*(-?\d+)$", line)
                            if match:
                                axis, imu_id, acc, gyro = match.groups()
                                # if imu_id == '3':
                                #     print(line)
                                if imu_id == '1':
                                    if axis == 'X':
                                        self.imu_dic['imu_1'].AccX, self.imu_dic['imu_1'].GyroX = (
                                            float(acc) / self.imu_dic['imu_1'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_1'].Acc_zero_bias[0],
                                            float(gyro) / self.imu_dic['imu_1'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_1'].Gyro_zero_bias[0])
                                    elif axis == 'Y':
                                        self.imu_dic['imu_1'].AccY, self.imu_dic['imu_1'].GyroY = (
                                            float(acc) / self.imu_dic['imu_1'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_1'].Acc_zero_bias[1],
                                            float(gyro) / self.imu_dic['imu_1'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_1'].Gyro_zero_bias[1])
                                    elif axis == 'Z':
                                        self.imu_dic['imu_1'].AccZ, self.imu_dic['imu_1'].GyroZ = (
                                            float(acc) / self.imu_dic['imu_1'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_1'].Acc_zero_bias[2],
                                            float(gyro) / self.imu_dic['imu_1'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_1'].Gyro_zero_bias[2])
                                elif imu_id == '2':
                                    if axis == 'X':
                                        self.imu_dic['imu_2'].AccX, self.imu_dic['imu_2'].GyroX = (
                                            float(acc) / self.imu_dic['imu_2'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_2'].Acc_zero_bias[0],
                                            float(gyro) / self.imu_dic['imu_2'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_2'].Gyro_zero_bias[0])
                                    elif axis == 'Y':
                                        self.imu_dic['imu_2'].AccY, self.imu_dic['imu_2'].GyroY = (
                                            float(acc) / self.imu_dic['imu_2'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_2'].Acc_zero_bias[1],
                                            float(gyro) / self.imu_dic['imu_2'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_2'].Gyro_zero_bias[1])
                                    elif axis == 'Z':
                                        self.imu_dic['imu_2'].AccZ, self.imu_dic['imu_2'].GyroZ = (
                                            float(acc) / self.imu_dic['imu_2'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_2'].Acc_zero_bias[2],
                                            float(gyro) / self.imu_dic['imu_2'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_2'].Gyro_zero_bias[2])

                                elif imu_id == '3':
                                    if axis == 'X':
                                        self.imu_dic['imu_3'].AccX, self.imu_dic['imu_3'].GyroX = (
                                            float(acc) / self.imu_dic['imu_3'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_3'].Acc_zero_bias[0],
                                            float(gyro) / self.imu_dic['imu_3'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_3'].Gyro_zero_bias[0])
                                    elif axis == 'Y':
                                        self.imu_dic['imu_3'].AccY, self.imu_dic['imu_3'].GyroY = (
                                            float(acc) / self.imu_dic['imu_3'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_3'].Acc_zero_bias[1],
                                            float(gyro) / self.imu_dic['imu_3'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_3'].Gyro_zero_bias[1])
                                    elif axis == 'Z':
                                        self.imu_dic['imu_3'].AccZ, self.imu_dic['imu_3'].GyroZ = (
                                            float(acc) / self.imu_dic['imu_3'].ACC_SENSITIVITY * GRAVITY -
                                            self.imu_dic['imu_3'].Acc_zero_bias[2],
                                            float(gyro) / self.imu_dic['imu_3'].GYRO_SENSITIVITY * DEG_TO_RAD -
                                            self.imu_dic['imu_3'].Gyro_zero_bias[2])

                with imu_lock:
                    self.imu_dic['imu_1'].imu_np = np.array(
                        [self.imu_dic['imu_1'].AccX, self.imu_dic['imu_1'].AccY, self.imu_dic['imu_1'].AccZ,
                         self.imu_dic['imu_1'].GyroX, self.imu_dic['imu_1'].GyroY,
                         self.imu_dic['imu_1'].GyroZ])
                    self.imu_dic['imu_2'].imu_np = np.array(
                        [self.imu_dic['imu_2'].AccX, self.imu_dic['imu_2'].AccY, self.imu_dic['imu_2'].AccZ,
                         self.imu_dic['imu_2'].GyroX, self.imu_dic['imu_2'].GyroY,
                         self.imu_dic['imu_2'].GyroZ])
                    self.imu_dic['imu_3'].imu_np = np.array(
                        [self.imu_dic['imu_3'].AccX, self.imu_dic['imu_3'].AccY, self.imu_dic['imu_3'].AccZ,
                         self.imu_dic['imu_3'].GyroX, self.imu_dic['imu_3'].GyroY,
                         self.imu_dic['imu_3'].GyroZ])

                    # Calculate Quaternion
                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    for i in range(1, 4):
                        self.imu_dic[f'imu_{i}'].gyro_to_quat(dt)

                        self.imu_dic[f'imu_{i}'].Quat_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].Quat
                        self.imu_dic[f'imu_{i}'].imu_data_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].imu_np

                (self.imu_dic['imu_1'].AccX, self.imu_dic['imu_1'].AccY, self.imu_dic['imu_1'].AccZ,
                 self.imu_dic['imu_1'].GyroX, self.imu_dic['imu_1'].GyroY, self.imu_dic['imu_1'].GyroZ) = 0, 0, 0, 0, 0, 0
                (self.imu_dic['imu_2'].AccX, self.imu_dic['imu_2'].AccY, self.imu_dic['imu_2'].AccZ,
                 self.imu_dic['imu_2'].GyroX, self.imu_dic['imu_2'].GyroY, self.imu_dic['imu_2'].GyroZ) = 0, 0, 0, 0, 0, 0
                (self.imu_dic['imu_3'].AccX, self.imu_dic['imu_3'].AccY, self.imu_dic['imu_3'].AccZ,
                 self.imu_dic['imu_3'].GyroX, self.imu_dic['imu_3'].GyroY, self.imu_dic['imu_3'].GyroZ) = 0, 0, 0, 0, 0, 0
                self.COUNT_NUM += 1

        except KeyboardInterrupt:
            print("Exiting...")

        finally:
            if self.serial and self.serial.is_open and self.main_thread_flag:
                self.serial.close()
                print('serial is closed')

    def stop_collect(self):
        if not self.is_collecting or self.t1 is None:
            print("There is no thread running")
            return

        self.is_collecting = False  # It is important
        self.stop_event.set()

        self.t1.join(timeout=1)

        if self.t1.is_alive():
            print("Warning: t1 may be still running")
        self.t1 = None

    def initial_Gyro_calibration(self, calibration_duration, preparation_time):
        whether_calibration = input('Do you want to calibrate *** the Gyro Zero Bias *** ? (Y/n)\n')

        if whether_calibration == 'Y':
            print(f'You have {preparation_time}s to prepare')

            for i in range(preparation_time, 1, -1):
                print(f"{i}s left")
                time.sleep(0.99)

            for i in range(1, 4):
                self.imu_dic[f'imu_{i}'].GyroX_zero_bias = 0
                self.imu_dic[f'imu_{i}'].GyroY_zero_bias = 0
                self.imu_dic[f'imu_{i}'].GyroZ_zero_bias = 0
                self.imu_dic[f'imu_{i}'].Gyro_zero_bias = np.array([0, 0, 0])

            self.is_collecting = True

            print(' ---- Calibration Start ---- ')
            self.t1 = threading.Thread(target=self.collect_imu_raw_data, daemon=True)
            self.t1.start()
            time.sleep(calibration_duration)

            print(self.imu_dic['imu_1'].imu_data_df.shape)
            self.imu_dic['imu_1'].imu_data_df.to_csv('DataCSV/test_imu.csv', index=False)

            for i in range(1, 4):
                self.imu_dic[f'imu_{i}'].calculate_mean_of_data()
                print(f"the *PREVIOUS* Gyro Zero Bias are:\n"
                      f"GyroX Bias: {self.imu_dic[f'imu_{i}'].GyroX_zero_bias}\n"
                      f"GyroY Bias: {self.imu_dic[f'imu_{i}'].GyroY_zero_bias}\n"
                      f"GyroZ Bias: {self.imu_dic[f'imu_{i}'].GyroZ_zero_bias}\n")
                print(f"the *CURRENT* Gyro Zero Bias are:\n"
                      f"New GyroX Bias: {self.imu_dic[f'imu_{i}'].mean_GyroX}\n"
                      f"New GyroY Bias: {self.imu_dic[f'imu_{i}'].mean_GyroY}\n"
                      f"New GyroZ Bias: {self.imu_dic[f'imu_{i}'].mean_GyroZ}\n")
                whether_replace = input(f"Do you want to replace all the Gyro Bias of imu_{i}? (Y/n)\n")

                if whether_replace:
                    with open('config.json', 'r') as f:
                        data = json.load(f)

                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroX_zero_bias'] = self.imu_dic[f'imu_{i}'].mean_GyroX
                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroY_zero_bias'] = self.imu_dic[f'imu_{i}'].mean_GyroY
                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroZ_zero_bias'] = self.imu_dic[f'imu_{i}'].mean_GyroZ

                    with open('config.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)  # indent=2 保留格式化，方便阅读

            # reset all imu data
            self.imu_dic['imu_1'].reset()
            self.imu_dic['imu_2'].reset()
            self.imu_dic['imu_3'].reset()
            self.COUNT_NUM = 0

            self.stop_collect()

        elif whether_calibration == 'n':
            return
        else:
            print("Please input 'Y' or 'n'")
            print("Whatever, I just ignore it...")
            # self.initial_Gyro_calibration(calibration_duration)

    def start_threading_collect(self, main_threading_time):

        self.is_collecting = True
        self.main_thread_flag = True
        self.stop_event.clear()

        print('The main thread is starting')
        self.t1 = threading.Thread(target=self.collect_imu_raw_data, daemon=True)
        self.t1.start()

        time.sleep(main_threading_time)

        self.stop_collect()



SAMPLING_FREQUENCY = config['serial']['SAMPLING_RATE']
GRAVITY = config['mpu6050_1']['Acc_params']['GRAVITY']  # m/s²
DEG_TO_RAD = math.pi / 180.0
imu_lock = threading.Lock()

if __name__ == "__main__":

    # 配置串口
    ser = serial.Serial(
        port=config['serial']['port'],  # 根据实际情况修改端口号
        baudrate=config['serial']['baud_rate'],  # 根据实际情况修改波特率
        timeout=config['serial']['timeout']
    )

    # Initiate IMU Params
    COUNT_NUM = 0
    imu_1 = IMUData('mpu6050_1')
    imu_2 = IMUData('mpu6050_2')
    imu_3 = IMUData('mpu6050_3')

    three_imu_collector = ThreeIMUCollector(ser, imu_1, imu_2, imu_3, count_num=COUNT_NUM, duration=100)

    # Calibration of Gyro
    CALIBRATION_TIME = 20
    preparation_time = 1
    three_imu_collector.initial_Gyro_calibration(calibration_duration=CALIBRATION_TIME,
                                                 preparation_time=preparation_time)

    # Main Thread
    MAIN_THREAD_DURATION = 5
    three_imu_collector.start_threading_collect(main_threading_time=MAIN_THREAD_DURATION)

    print(imu_1.imu_data_df.shape)
    print(imu_2.imu_data_df.shape)
    print(imu_3.imu_data_df.shape)

    imu_1.save_as_csv('DataCSV/imu_1/imu_threading_data.csv')
    imu_2.save_as_csv('DataCSV/imu_2/imu_threading_data.csv')
    imu_3.save_as_csv('DataCSV/imu_3/imu_threading_data.csv')

    imu_1.Quat_df.to_csv("DataCSV/imu_1/imu_threading_Quat.csv")
    imu_2.Quat_df.to_csv("DataCSV/imu_2/imu_threading_Quat.csv")
    imu_3.Quat_df.to_csv("DataCSV/imu_3/imu_threading_Quat.csv")


    # plot_histogram(imu_1.imu_data_df)
