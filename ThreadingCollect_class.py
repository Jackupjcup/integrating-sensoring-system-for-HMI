import numpy as np
import serial
import re
import time
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
import math
import threading
import json

imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

SAMPLING_FREQUENCY = imu_config['serial']['SAMPLING_RATE']
GRAVITY = imu_config['mpu6050_1']['Acc_params']['GRAVITY']  # m/s²
DEG_TO_RAD = math.pi / 180.0
imu_lock = threading.Lock()
PnF_lock = threading.Lock()

class ThreadingCollect:
    def __init__(self, ser, imu_1, imu_2, imu_3, palm_points, duration):
        self.imu_dic = {'imu_1': imu_1,
                        'imu_2': imu_2,
                        'imu_3': imu_3}
        self.palm_points = palm_points
        self.COUNT_NUM = 0
        self.duration = duration
        self.serial = ser
        self.imu_dt_list = []
        self.palm_points_dt_list = []

        self.t1 = None
        self.Start_time = time.time()

        self.is_collecting = False
        self.main_thread_flag = False
        self.stop_event = threading.Event()
        
        # Palm Points Parameters
        self.palm_points_is_capturing = False
        self.imu_lock = imu_lock
        self.PnF_lock = PnF_lock

        # IMU Parameters
        self.IMU_is_capturing = False
        self.capture_X1_counts = 0

        # Calibration Markers
        self.pressure_calibration_flag = False

    def init_Gyro_calibration(self, calibration_duration, preparation_time):
        whether_calibration = 'Y' #input('Do you want to calibrate *** the Gyro Zero Bias *** ? (Y/n)\n')

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

            print(' ----  Gyroscope Calibration Start ---- ')
            self.Start_time = time.time()
            self.t1 = threading.Thread(target=self.collecting, daemon=True)
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
                whether_replace = "Y"#input(f"Do you want to replace all the Gyro Bias of imu_{i}? (Y/n)\n")

                if whether_replace:
                    with open('D:/blender_workplace/modules/IMUProcessing/config.json', 'r') as f:
                        data = json.load(f)

                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroX_zero_bias'] = self.imu_dic[
                        f'imu_{i}'].mean_GyroX
                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroY_zero_bias'] = self.imu_dic[
                        f'imu_{i}'].mean_GyroY
                    data[self.imu_dic[f'imu_{i}'].imu_id_str]['Gyro_params']['GyroZ_zero_bias'] = self.imu_dic[
                        f'imu_{i}'].mean_GyroZ

                    with open('D:/blender_workplace/modules/IMUProcessing/config.json', 'w', encoding='utf-8') as f:
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

    def pressure_sensor_calibration(self, pressure_calibration_duration=5):
        self.pressure_calibration_flag = True
        self.is_collecting = True
        self.palm_points.whether_dynamic_adjusting = False
        self.palm_points.press_sensor_zero_bias_np = np.zeros(self.palm_points.points_num)

        if self.pressure_calibration_flag:
            self.palm_points.calibration_flag = True
            self.palm_points.whether_denoise = False

        print(' ---- Pressure Calibration Start ---- ')
        self.Start_time = time.time()
        self.t1 = threading.Thread(target=self.collecting, daemon=True)
        self.t1.start()
        time.sleep(pressure_calibration_duration)

        self.t1.join(timeout=0.1)

        self.palm_points.press_sensor_zero_bias_np = np.mean(self.palm_points.collected_points_df.values, axis=0)
        # pressure_points_std = np.std(np.array(self.palm_points.collected_points_df), axis=0)
        # print(pressure_points_std)

        with open('D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json', 'r') as f:
            data = json.load(f)
        data["PlaneHandPressure_Display"]["ZeroBias"] = list(self.palm_points.press_sensor_zero_bias_np)

        with open('D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # indent=2 保留格式化，方便阅读
        print(' ---- Pressure Calibration End ---- ')
        
    def collecting(self):
        print("Collecting...")
        temp_matrix = []
        pattern = re.compile(r'^([XYZ])(\d)A(-?\d+)G(-?\d+)$')

        while not self.stop_event.is_set():
            try:
                line_bytes = self.serial.readline()
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                match = pattern.match(line)
                # print(line)
            
            except Exception:
                continue
                
            if not line:
                continue

            if line == "FRAME_START":
                temp_matrix = []
                self.palm_points_is_capturing = True

            elif line == "FRAME_END":
                self.palm_points_is_capturing = False
                # 检查是否收齐了 14 行
                if len(temp_matrix) == self.palm_points.rows_num:
                    with self.PnF_lock:
                        try:
                            # 转换为 numpy 数组并发送
                            final_data = np.array(temp_matrix, dtype=float)
                            # print(final_data.shape)
                            self.palm_points.transform_latest_pressure_data(final_data)

                            if not self.palm_points_dt_list:
                                palm_points_last_time = time.time()
                                self.palm_points_dt_list.append(0.0)
                            else:
                                palm_points_now_time = time.time()
                                palm_points_dt = palm_points_now_time - self.Start_time
                                self.palm_points_dt_list.append(palm_points_dt)


                        except Exception as e:
                            print(f"数据转换错误: {e}")

            elif self.palm_points_is_capturing:
                # 解析具体数据行 "0 0 11 33..."
                try:
                    vals = [int(x) for x in line.split() if x.isdigit()]
                    if len(vals) == self.palm_points.cols_num:
                        temp_matrix.append(vals)
                except ValueError:
                    pass
##########################################################################################################################
            elif match and not self.IMU_is_capturing:
                axis, imu_id, acc, gyro = match.groups()
                if axis == "X" and imu_id == "1":
                    self.IMU_is_capturing = True
                    self.capture_X1_counts += 1

                    self.imu_dic['imu_1'].IMU_data_dic[f"Acc{axis}"], self.imu_dic['imu_1'].IMU_data_dic[f"Gyro{axis}"] = (
                        float(acc) / self.imu_dic['imu_1'].ACC_SENSITIVITY * GRAVITY - self.imu_dic['imu_1'].Acc_zero_bias[0],
                        float(gyro) / self.imu_dic['imu_1'].GYRO_SENSITIVITY * DEG_TO_RAD - self.imu_dic['imu_1'].Gyro_zero_bias[0])


            elif match and self.IMU_is_capturing: # Collect rest of IMU data except X1 acc and gyro
                axis, imu_id, acc, gyro = match.groups()
                if axis == "X":
                    zero_bias_idx = 0
                elif axis == "Y":
                    zero_bias_idx = 1
                else:
                    zero_bias_idx = 2
                self.imu_dic[f"imu_{imu_id}"].IMU_data_dic[f"Acc{axis}"], self.imu_dic[f"imu_{imu_id}"].IMU_data_dic[f"Gyro{axis}"] = (
                    float(acc) / self.imu_dic[f"imu_{imu_id}"].ACC_SENSITIVITY * GRAVITY - self.imu_dic[f"imu_{imu_id}"].Acc_zero_bias[zero_bias_idx],
                    float(gyro) / self.imu_dic[f"imu_{imu_id}"].GYRO_SENSITIVITY * DEG_TO_RAD - self.imu_dic[f'imu_{imu_id}'].Gyro_zero_bias[zero_bias_idx])


                if axis == "Z" and imu_id == "3":
                    self.IMU_is_capturing = False

                    if not self.imu_dt_list:
                        imu_dt = 0
                        imu_last_time = time.time()
                        self.imu_dt_list.append(0)
                    else:
                        imu_now_time = time.time()
                        imu_dt = imu_now_time - imu_last_time
                        imu_last_time = imu_now_time
                        self.imu_dt_list.append(imu_dt)

                    with imu_lock:
                        for i in range(1, 4):
                            self.imu_dic[f"imu_{i}"].collect_imu_data()
                            # 卡尔曼滤波更新角度
                            self.imu_dic[f'imu_{i}'].kalman_update_angles(imu_dt)

                            # self.imu_dic[f"imu_{i}"].imu_np = np.array(
                            #     [self.imu_dic[f"imu_{i}"].AccX, self.imu_dic[f"imu_{i}"].AccY, self.imu_dic[f"imu_{i}"].AccZ,
                            #      self.imu_dic[f"imu_{i}"].GyroX, self.imu_dic[f"imu_{i}"].GyroY, self.imu_dic[f"imu_{i}"].GyroZ])

                            # self.imu_dic[f'imu_{i}'].gyro_to_quat(gyro=self.imu_dic[f'imu_{i}'].get_kalman_angles().copy(),
                            #                                       dt=imu_dt)
                            self.imu_dic[f'imu_{i}'].gyro_to_quat(gyro=self.imu_dic[f'imu_{i}'].get_raw_euler_degrees().copy(),
                                                                  dt=imu_dt)
                            self.imu_dic[f'imu_{i}'].Quat_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].Quat
                            self.imu_dic[f'imu_{i}'].imu_data_df.loc[self.COUNT_NUM, :] = self.imu_dic[f'imu_{i}'].imu_np

                            self.imu_dic[f"imu_{i}"].clear_dic()

                        self.COUNT_NUM += 1




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
        
    def start_threading_collect(self, main_threading_time):

        self.is_collecting = True
        self.main_thread_flag = True
        self.stop_event.clear()

        print('The main thread is starting')
        self.Start_time = time.time()
        self.t1 = threading.Thread(target=self.collecting, daemon=True)
        self.t1.start()

        time.sleep(main_threading_time) # NOT GOOD, NEED TO BE CHANGED

        self.stop_collect()





if __name__ == '__main__':
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
    palm_points = PalmPointsClass(14, 10, "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")


    threading_collect = ThreadingCollect(ser=ser,
                                         imu_1=imu_1,
                                         imu_2=imu_2,
                                         imu_3=imu_3,
                                         palm_points=palm_points,
                                         duration=500)

    '''Calibration'''
    # threading_collect.init_Gyro_calibration(10, preparation_time=1)  # IMU Calibration
    # threading_collect.pressure_sensor_calibration() # Pressure Sensor Calibration

    '''Main Thread Collecting'''
    threading_collect.start_threading_collect(main_threading_time=5)

    print(f"num of IMU is {imu_1.imu_data_df.shape}")
    # print(imu_2.imu_data_df.shape)
    # print(imu_3.imu_data_df.shape)
    print(f"shape of pressure data: {palm_points.collected_points_df.shape}")

    print(len(threading_collect.imu_dt_list))
    # print(np.mean(threading_collect.imu_dt_list))
    print(len(threading_collect.palm_points_dt_list))
    # print(threading_collect.palm_points_dt_list)
    # print(np.mean(threading_collect.palm_points_dt_list))

    np.save("D:/blender_workplace/modules/PalmPoints_and_FingerBend/pressure_data_sequence.npy", palm_points.pressure_data_sequence)

    # imu_1.save_as_csv("DataCSV/imu_1/raw_data.csv")
    # imu_1.Quat_df.to_csv("DataCSV/imu_1/Quat.csv", index=False)
    # imu_2.save_as_csv("DataCSV/imu_2/raw_data.csv")
    # imu_2.Quat_df.to_csv("DataCSV/imu_2/Quat.csv", index=False)
    # imu_3.save_as_csv("DataCSV/imu_3/raw_data.csv")
    # imu_3.Quat_df.to_csv("DataCSV/imu_3/Quat.csv", index=False)

















