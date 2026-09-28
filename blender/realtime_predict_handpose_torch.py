import numpy as np
import threading
import sys
import os
import time
# PROJECT_ROOT = "D:/blender_workplace/modules"
#
# # 将项目根目录加入Python搜索路径（确保能找到所有子模块）
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
#     print(f"已添加项目路径到sys.path：{PROJECT_ROOT}")

# 导入自定义模型和数据处理类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入传感器相关模块
from ThreadingCollect_class import ThreadingCollect
from IMUProcessing.MPU6050_utils import load_config, IMUData
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
from hand_pose_mediapipe.smallCNN_regression import SmallCNN_regression
import serial
import torch


# 常量设置
ROWS, COLS = 14, 10
TOTAL_NUM = ROWS * COLS
UPDATE_INTERVAL = 100
MAX_VALUE = 500

def load_model():
    '''Model Parameters'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载模型
    model = SmallCNN_regression(num_classes=14).to(device)
    model_path = 'D:/blender_workplace/modules/hand_pose_mediapipe/train/best_cnn_regression_model.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from epoch {checkpoint["epoch"] + 1}')
        print(f'Model validation loss: {checkpoint["val_loss"]:.6f}')
    else:
        print(f'Warning: Model file {model_path} not found!')
        print('Using untrained model...')

    model.eval()  # 设置为评估模式

    # 加载标准化参数（从训练集中获取）
    try:
        # 尝试加载训练时保存的标准化参数
        results_path = 'D:/blender_workplace/modules/hand_pose_mediapipe/train/training_results.npy'
        if os.path.exists(results_path):
            results = np.load(results_path, allow_pickle=True).item()
            print('Loaded normalization parameters from training results')
        else:
            # 如果没有保存的参数，使用默认值
            pressure_mean = 0.0
            pressure_std = 1.0
            print('Using default normalization parameters')
    except:
        pressure_mean = 0.0
        pressure_std = 1.0
        print('Failed to load normalization parameters, using defaults')

    return model, device

class PredictHandPose:
    def __init__(self, thread_collect):
        self.model, self.device = load_model()
        self.model.eval()

        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threading_collect = thread_collect
        self.predicted_angles_np = None



    def predict_hand_pose(self,):
        self.stop_event.clear()
        self._threads = []
        self._threads.append(threading.Thread(target=self.threading_collect.collecting))
        for t in self._threads:
            t.daemon = True
            t.start()


        try:
            while not self.stop_event.is_set():
                with self.data_lock:
                    # 获取最新压力数据
                    if hasattr(self.threading_collect, 'palm_points') and self.threading_collect.palm_points.latest_pressure_data is not None:
                        raw_data = self.threading_collect.palm_points.latest_pressure_data
                        # print(raw_data.shape)
                    else:
                        raw_data = np.zeros(TOTAL_NUM)  # 无数据时显示0

                    with torch.no_grad():
                        # 预处理压力数据
                        if np.sum(np.abs(raw_data)) > 0:  # 如果有有效数据
                            # 重塑为14x10
                            pressure_data = raw_data.reshape(ROWS, COLS).astype(np.float32)

                            # 转换为torch张量
                            pressure_tensor = torch.from_numpy(pressure_data).unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 10)
                            pressure_tensor = pressure_tensor.to(self.device)

                            # 模型预测
                            predicted_angles = self.model(pressure_tensor)
                            predicted_angles_np = predicted_angles.cpu().numpy().flatten()
                            self.predicted_angles_np = np.clip(predicted_angles_np, a_min=0, a_max=1) * 180
                        else:
                            self.predicted_angles_np = np.zeros(14)

        finally:
            self.stop_event.set()
            for t in self._threads:
                if t.is_alive():
                    t.join(timeout=1.0)  # 等待1秒，超时强制终止
            print("采集线程和预测循环已停止")

    def stop_predict(self):
        self.stop_event.set()


if __name__ == "__main__":
    imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
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

    predict_thread = PredictHandPose(threading_collect)

    predict_thread.predict_hand_pose()











