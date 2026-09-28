from imu_threading import ThreeIMUCollector
import serial
from MPU6050_utils import load_config, IMUData

#
config = load_config()


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
    # three_imu_collector.initial_Gyro_calibration(calibration_duration=CALIBRATION_TIME,
    #                                              preparation_time=preparation_time)

    # Main Thread
    MAIN_THREAD_DURATION = 5
    three_imu_collector.start_threading_collect(main_threading_time=MAIN_THREAD_DURATION)

    print(imu_1.imu_data_df.shape)
    print(imu_2.imu_data_df.shape)
    print(imu_3.imu_data_df.shape)

    imu_1.save_as_csv('DataCSV/imu_1/imu_threading_data.csv')
    imu_2.save_as_csv('DataCSV/imu_2/imu_threading_data.csv')
    imu_3.save_as_csv('DataCSV/imu_3/imu_threading_data.csv')