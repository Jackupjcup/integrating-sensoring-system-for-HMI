import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import threading
import queue
import time
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from IMUProcessing.MPU6050_utils import load_config
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass, load_config as palm_load_config
import serial
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免与 OpenCV 冲突
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

class AngleTemporalFilter:
    def __init__(self, window_size=5):
        """
        时间平滑器

        参数
        ----
        window_size : 平滑窗口大小
        """
        self.window_size = window_size
        self.buffer = {}

    def update(self, angles_dict):
        """
        输入当前帧角度，返回平滑后的角度
        """
        smoothed = {}
        for finger, joints in angles_dict.items():
            if finger not in self.buffer:
                self.buffer[finger] = {}
            smoothed[finger] = {}
            for joint, angle in joints.items():
                if joint not in self.buffer[finger]:
                    self.buffer[finger][joint] = deque(maxlen=self.window_size)
                if angle is not None:
                    self.buffer[finger][joint].append(angle)
                buf = self.buffer[finger][joint]
                if len(buf) == 0:
                    smoothed_angle = None
                else:
                    smoothed_angle = float(np.mean(buf))
                smoothed[finger][joint] = smoothed_angle

        return smoothed

# ============================================================================
# 手指关节角度计算函数（MediaPipe版本）
# ============================================================================

def calculate_finger_angles_mediapipe(hand_landmarks, conf_threshold=0.5):
    """
    计算每个手指关节的弯曲角度（基于MediaPipe手部关键点）。

    参数:
        hand_landmarks: MediaPipe手部关键点列表，每个关键点有x, y, z属性（归一化坐标0-1）
        conf_threshold: 置信度阈值（MediaPipe没有每个关键点的置信度，但可以使用可见性）

    返回:
        angles_dict: 字典，包含每个手指每个关节的角度（单位：度）
            例如: {'thumb': {'cmc_P_M': 120.5, 'P_M_I': 110.2},
                  'index': {'wrist_P_M': 150.2, 'P_M_I': 130.1, 'M_I_tip': 100.5},
                  ...}
    """
    import math

    # MediaPipe手部关键点索引（共21个点）
    # 0: 手腕
    # 1: 拇指根部（CMC）
    # 2: 拇指掌指关节（MCP）
    # 3: 拇指指间关节（IP）
    # 4: 拇指指尖
    # 5: 食指根部（MCP）
    # 6: 食指近端指间关节（PIP）
    # 7: 食指远端指间关节（DIP）
    # 8: 食指指尖
    # 9: 中指根部（MCP）
    # 10: 中指近端指间关节（PIP）
    # 11: 中指远端指间关节（DIP）
    # 12: 中指指尖
    # 13: 无名指根部（MCP）
    # 14: 无名指近端指间关节（PIP）
    # 15: 无名指远端指间关节（DIP）
    # 16: 无名指指尖
    # 17: 小指根部（MCP）
    # 18: 小指近端指间关节（PIP）
    # 19: 小指远端指间关节（DIP）
    # 20: 小指指尖

    # 定义手指关键点索引组（每个手指三个关节角度）
    finger_groups = {
        'thumb': [(1, 2, 3), (2, 3, 4)],  # cmc-P-M, P-M-I (cmc-mcp-ip, mcp-ip-tip)
        'index': [(0, 5, 6), (5, 6, 7), (6, 7, 8)],  # wrist-P-M, P-M-I, M-I-tip
        'middle': [(0, 9, 10), (9, 10, 11), (10, 11, 12)],  # wrist-P-M, P-M-I, M-I-tip
        'ring': [(0, 13, 14), (13, 14, 15), (14, 15, 16)],  # wrist-P-M, P-M-I, M-I-tip
        'pinky': [(0, 17, 18), (17, 18, 19), (18, 19, 20)]  # wrist-P-M, P-M-I, M-I-tip
    }

    # 定义角度名称（使用简化的关节命名：P=靠近手掌的关节，M=中间关节，I=指尖关节）
    angle_names = {
        'thumb': ['P', 'M'],  # cmc-mcp-ip, mcp-ip-tip
        'index': ['P', 'M', 'I'],  # wrist-mcp-pip, mcp-pip-dip, pip-dip-tip
        'middle': ['P', 'M', 'I'],
        'ring': ['P', 'M', 'I'],
        'pinky': ['P', 'M', 'I']
    }

    # 检查手部关键点是否为空
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return {finger: {name: None for name in angle_names[finger]} for finger in angle_names}

    # MediaPipe关键点没有置信度，但可能有可见性属性（如果可用）
    # 这里我们假设所有关键点都是可见的

    angles_dict = {}

    for finger, groups in finger_groups.items():
        finger_angles = {}
        for angle_name, (i, j, k) in zip(angle_names[finger], groups):
            # 提取坐标（归一化坐标0-1）
            p1 = (hand_landmarks[i].x, hand_landmarks[i].y)
            p2 = (hand_landmarks[j].x, hand_landmarks[j].y)
            p3 = (hand_landmarks[k].x, hand_landmarks[k].y)

            # 计算向量
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # 计算角度（0-180度）
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if norm1 == 0 or norm2 == 0:
                finger_angles[angle_name] = None
                continue

            cos_angle = dot_product / (norm1 * norm2)
            # 防止数值误差
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))

            finger_angles[angle_name] = angle

        angles_dict[finger] = finger_angles

    return angles_dict


# ============================================================================
# 手指角度收集器类
# ============================================================================

class FingerAngleAndPressureSensorCollector:
    """
    手指角度收集器类，用于实时收集手指角度数据并保存到record文件夹
    """

    def __init__(self, record_dir="record", camera_id=0, use_threading=True, num_hands=2,
                 model_asset_path='hand_landmarker.task',
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 video_codec='avi', video_fps=30.0,
                 whether_collecting_pressure_data=True):
        """
        初始化手指角度收集器

        参数:
            record_dir: 记录目录路径
            camera_id: 摄像头ID
            use_threading: 是否使用线程
            num_hands: 最大手部数量
            model_asset_path: MediaPipe模型文件路径
            min_hand_detection_confidence: 手部检测置信度阈值
            min_hand_presence_confidence: 手部存在置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
            video_codec: 视频编码器 (例如: 'mp4v', 'XVID', 'MJPG')
            video_fps: 视频帧率
        """
        self.record_dir = Path(record_dir)
        self.camera_id = camera_id
        self.use_threading = use_threading
        self.num_hands = num_hands
        self.model_asset_path = model_asset_path
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.video_codec = video_codec
        self.video_fps = video_fps

        # 状态变量
        self.is_recording = False
        self.record_session_id = None
        self.video_writer = None
        self.data_file = None
        self.csv_writer = None
        self.start_time = None

        # 视频相关
        self.frame_width = None
        self.frame_height = None

        # 线程相关
        self.frame_queue = None
        self.result_queue = None
        self.stop_event = None
        self.threads = None
        self.detector = None

        # Pressure Sensor Array
        self.num_row = 14
        self.num_col = 10
        self.imu_config = load_config("D:/blender_workplace/modules/IMUProcessing/config.json")
        self.palm_config_path = "D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json"
        self.pressure_collector = None
        self.palm_points_is_capturing = False
        self.pressure_lock = threading.Lock()
        self.whether_collecting_pressure_data = whether_collecting_pressure_data

        # Smooth Angle Data
        self.angle_filter = AngleTemporalFilter(window_size=5)

        # Serial
        self.serial = None

        # 帧索引计数器
        self.frame_index = 0

        # 创建记录目录
        self.record_dir.mkdir(parents=True, exist_ok=True)

        print(f"手指角度收集器已初始化，记录目录: {self.record_dir.absolute()}")

    def _generate_session_id(self):
        """生成唯一的会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"

    def _setup_recording(self):
        """设置录制会话"""
        self.record_session_id = self._generate_session_id()
        session_dir = self.record_dir / self.record_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 创建数据文件 (CSV格式)
        data_file_path = session_dir / "angles_data.csv"
        self.data_file = open(data_file_path, 'w', newline='', encoding='utf-8')

        # Init Pressure Data CSV      It is not included in session_info.json
        pressure_data_file_path = session_dir / "pressure_data.csv"
        self.pressure_data_file = open(pressure_data_file_path, 'w', newline='', encoding='utf-8')
        pressure_csv_column = ['timestamp'] + [i for i in range(140)]
        self.pressure_csv_writer = csv.DictWriter(self.pressure_data_file, fieldnames=pressure_csv_column)

        # 定义CSV列
        fieldnames = ['timestamp', 'frame_index', 'hand_index', 'hand_label']

        # 添加角度列
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger in fingers:
            # 根据手指确定关节数量
            if finger == 'thumb':
                joints = ['P', 'M']
            else:
                joints = ['P', 'M', 'I']
            for joint in joints:
                fieldnames.append(f"{finger}_{joint}")

        self.csv_writer = csv.DictWriter(self.data_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

        # 创建视频文件路径
        video_file_path = session_dir / "video.avi"  # 使用.avi扩展名确保兼容性

        # 初始化视频写入器 (稍后在获取帧尺寸后设置)
        self.video_file_path = video_file_path

        # 保存会话信息
        session_info = {
            'session_id': self.record_session_id,
            'start_time': datetime.now().isoformat(),
            'camera_id': self.camera_id,
            'video_codec': self.video_codec,
            'video_fps': self.video_fps,
            'num_hands': self.num_hands,
            'fieldnames': fieldnames
        }

        info_file_path = session_dir / "session_info.json"
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        print(f"录制会话已创建: {self.record_session_id}")
        print(f"数据文件: {data_file_path}")
        print(f"视频文件: {video_file_path}")
        print(f"视频编码器: {self.video_codec}, 帧率: {self.video_fps}")

        # 如果帧尺寸已知，立即初始化视频写入器
        if self.frame_width is not None and self.frame_height is not None:
            print(f"帧尺寸已知 ({self.frame_width}x{self.frame_height})，初始化视频写入器...")
            self._init_video_writer(self.frame_width, self.frame_height)
        else:
            print("帧尺寸未知，视频写入器将在第一帧到达时延迟初始化")

    def _init_video_writer(self, frame_width, frame_height):
        """初始化视频写入器"""
        if self.video_writer is not None:
            print("视频写入器已存在，跳过初始化")
            return

        if self.video_file_path is None:
            print("错误: 视频文件路径未设置")
            return

        # 尝试的编码器列表（按优先级）
        codecs_to_try = []
        if hasattr(self, 'video_codec') and self.video_codec:
            codecs_to_try.append(self.video_codec)

        # 添加备用编码器
        codecs_to_try.extend(['XVID', 'MJPG', 'mp4v', 'DIVX'])

        # 去重
        codecs_to_try = list(dict.fromkeys(codecs_to_try))

        video_writer = None
        successful_codec = None

        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(
                    str(self.video_file_path),
                    fourcc,
                    self.video_fps,
                    (frame_width, frame_height)
                )

                if video_writer.isOpened():
                    successful_codec = codec
                    print(f"视频写入器初始化成功，编码器: {codec}, 尺寸: {frame_width}x{frame_height}")
                    break
                else:
                    video_writer = None
                    print(f"编码器 {codec} 不可用，尝试下一个...")
            except Exception as e:
                print(f"使用编码器 {codec} 时出错: {e}")
                video_writer = None
                continue

        self.video_writer = video_writer

        if self.video_writer is None:
            print("错误: 所有编码器都失败，视频将无法保存")
            print(f"尝试的编码器: {codecs_to_try}")
            print(f"视频路径: {self.video_file_path}")

    def _record_angles(self, timestamp, frame_index, hand_index, hand_label, angles):
        """记录角度数据到CSV"""
        if not self.csv_writer or not self.is_recording:
            return

        # 创建行数据
        row_data = {
            'timestamp': timestamp,
            'frame_index': frame_index,
            'hand_index': hand_index,
            'hand_label': hand_label
        }

        # 添加所有角度
        for finger, finger_angles in angles.items():
            for joint, angle in finger_angles.items():
                column_name = f"{finger}_{joint}"
                row_data[column_name] = angle

        # 写入行
        self.csv_writer.writerow(row_data)
        self.data_file.flush()

    def _record_pressure(self, timestamp, pressure_data_flatten):
        if not self.pressure_csv_writer or not self.is_recording:
            print("in record process but wrong")
            return

        pressure_dict = {idx: val for idx, val in enumerate(pressure_data_flatten)}
        row_data = {
            'timestamp': timestamp
        }
        row_data.update(pressure_dict)

        self.pressure_csv_writer.writerow(row_data)
        self.pressure_data_file.flush()



    def _record_video_frame(self, frame):
        """记录视频帧"""
        if not self.is_recording:
            return

        # 如果视频写入器未初始化，尝试初始化
        if self.video_writer is None:
            # 优先使用存储的帧尺寸
            if self.frame_width is not None and self.frame_height is not None:
                self._init_video_writer(self.frame_width, self.frame_height)
                print(f"视频写入器延迟初始化，使用存储的尺寸: {self.frame_width}x{self.frame_height}")
            # 备选方案：使用当前帧的尺寸
            elif frame is not None and len(frame.shape) >= 2:
                height, width = frame.shape[:2]
                self._init_video_writer(width, height)
                print(f"视频写入器延迟初始化，使用当前帧尺寸: {width}x{height}")
            else:
                print("警告: 无法初始化视频写入器，没有可用的帧尺寸")
                return

        # 写入帧
        if self.video_writer:
            self.video_writer.write(frame)

    def _cleanup_recording(self):
        """清理录制资源"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("视频写入器已释放")

        if self.data_file:
            self.data_file.close()
            self.data_file = None
            self.csv_writer = None
            print("手指角度数据文件已关闭")

        if self.pressure_data_file:
            self.pressure_data_file.close()
            self.pressure_data_file = None
            self.pressure_csv_writer = None
            print("压力传感器数据文件已关闭")

        # 保存结束时间
        if self.record_session_id:
            session_dir = self.record_dir / self.record_session_id
            info_file_path = session_dir / "session_info.json"

            if info_file_path.exists():
                with open(info_file_path, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)

                session_info['end_time'] = datetime.now().isoformat()
                session_info['duration_seconds'] = (datetime.fromisoformat(session_info['end_time']) -
                                                    datetime.fromisoformat(session_info['start_time'])).total_seconds()

                with open(info_file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_info, f, indent=2, ensure_ascii=False)

        self.is_recording = False
        print("录制已停止")

    def start_recording(self):
        """开始录制"""
        if self.is_recording:
            print("录制已经在进行中")
            return

        self._setup_recording()
        self.is_recording = True
        self.start_time = time.time()
        print("开始录制...")

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            print("没有正在进行的录制")
            return

        self._cleanup_recording()

    # ============================================================================
    # 线程函数（不再嵌套，便于单独调用）
    # ============================================================================

    def collect_pressure_data(self):
        """压力数据收集线程函数"""
        # Through Serial Collecting Pressure Data
        self.serial = serial.Serial(port=self.imu_config['serial']['port'],  # 根据实际情况修改端口号
                                    baudrate=self.imu_config['serial']['baud_rate'],  # 根据实际情况修改波特率
                                    timeout=self.imu_config['serial']['timeout']
                                    )

        self.pressure_collector = PalmPointsClass(rows_number=self.num_row,
                                                  cols_number=self.num_col,
                                                  config_path=self.palm_config_path)
        temp_matrix = []
        while not self.stop_event.is_set():
            try:
                line_bytes = self.serial.readline()
                line = line_bytes.decode("utf-8", errors="ignore").strip()

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
                if len(temp_matrix) == self.pressure_collector.rows_num:
                    with self.pressure_lock:
                        try:
                            # 转换为 numpy 数组并发送
                            final_data = np.array(temp_matrix, dtype=float)  # shape of (14, 10)
                            self.pressure_collector.transform_latest_pressure_data(final_data)
                            # final_data_flatten = final_data.flatten()
                            if self.is_recording:
                                timestamp = time.perf_counter()
                                self._record_pressure(timestamp, self.pressure_collector.latest_pressure_data)

                        except Exception as e:
                            print(f"数据转换错误: {e}")

            elif self.palm_points_is_capturing:
                # 解析具体数据行 "0 0 11 33..."
                try:
                    vals = [int(x) for x in line.split() if x.isdigit()]
                    if len(vals) == self.pressure_collector.cols_num:
                        temp_matrix.append(vals)
                except ValueError:
                    pass
            else:
                pass

    def capture_frames(self):
        """摄像头捕获线程函数"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 (ID: {self.camera_id})")
            print("请检查摄像头是否连接或被其他程序占用")
            # 放入一个None帧表示错误
            self.frame_queue.put(None)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"摄像头已打开 (ID: {self.camera_id}), 分辨率: 640x480")

        # 获取帧尺寸用于视频写入器
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = width
        self.frame_height = height
        print(f"摄像头分辨率: {width}x{height}")

        # 如果已经在录制中，立即初始化视频写入器
        if self.is_recording and self.video_file_path and not self.video_writer:
            self._init_video_writer(width, height)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            # 将帧放入队列（如果队列已满，丢弃旧帧）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame.copy())

        cap.release()
        print("摄像头已释放")

    def inference_worker(self):
        """推理线程函数"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取帧
                frame = self.frame_queue.get(timeout=0.1)

                # 检查帧是否有效
                if frame is None:
                    print("推理线程: 收到无效帧 (None)")
                    continue

                # 转换BGR到RGB（MediaPipe使用RGB）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # 推理
                detection_result = self.detector.detect(mp_image)

                # 将结果放入队列
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.result_queue.put((frame, detection_result, self.frame_index))
                self.frame_index += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"推理错误: {e}")
                import traceback
                traceback.print_exc()  # 打印详细堆栈信息
                continue

    def display_worker(self):
        """显示和角度计算线程函数"""
        # 初始化压力热力图的 matplotlib figure（只创建一次）
        palm_config = palm_load_config(self.palm_config_path)
        pressure_coords = palm_config["PlaneHandPressure_Display"]["PlaneCoords"]

        # 计算坐标范围
        x_coords = [coord[0] for coord in pressure_coords]
        y_coords = [coord[1] for coord in pressure_coords]
        x_min, x_max = min(x_coords) - 50, max(x_coords) + 50
        y_min, y_max = min(y_coords) - 50, max(y_coords) + 50

        # 创建 figure
        self.pressure_fig, self.pressure_ax = plt.subplots(figsize=(5, 5))
        self.pressure_fig.tight_layout(pad=1.0)

        # 初始 scatter plot
        initial_pressure = np.zeros(140)
        self.pressure_scatter = self.pressure_ax.scatter(
            x_coords,
            [-y for y in y_coords],  # y轴翻转
            c=initial_pressure,
            cmap='coolwarm',
            vmin=0,
            vmax=500,
            marker='s',
            s=100,
            edgecolors='white',
            linewidth=0.5,
        )
        self.pressure_ax.set_xlim(x_min, x_max)
        self.pressure_ax.set_ylim(-y_max, -y_min)  # y轴翻转后的范围
        self.pressure_ax.set_aspect('equal')
        self.pressure_ax.set_title('Pressure Heatmap', fontsize=10)
        self.pressure_ax.axis('off')

        # 存储坐标用于更新
        self.pressure_coords = pressure_coords

        while not self.stop_event.is_set():
            try:
                frame, detection_result, current_frame_index = self.result_queue.get(timeout=0.1)

                if frame is None or detection_result is None:
                    continue

                display_frame = frame.copy()

                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness

                # BGR -> RGB (只转换一次)
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # 存储每只手的信息（标签和角度）
                hand_info_list = []

                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    hand_label = handedness_list[idx][0].category_name

                    # 绘制关键点
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 计算角度
                    angles = calculate_finger_angles_mediapipe(hand_landmarks)
                    smoothed_angles = self.angle_filter.update(angles)

                    # 记录角度数据（如果正在录制）
                    if self.is_recording:
                        timestamp = time.perf_counter()
                        self._record_angles(timestamp, current_frame_index, idx, hand_label, smoothed_angles)

                    # 存储手部信息
                    hand_info_list.append({
                        'hand_label': hand_label,
                        'angles': smoothed_angles,
                        'hand_index': idx
                    })

                # 记录视频帧（如果正在录制）
                if self.is_recording:
                    self._record_video_frame(frame)

                # RGB -> BGR
                display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # 在转换后的图像上绘制文本
                for idx, hand_info in enumerate(hand_info_list):
                    hand_label = hand_info['hand_label']
                    smoothed_angles = hand_info['angles']

                    y_offset = 30 + idx * 150

                    # 绘制手部标签和录制状态
                    status_text = "recording" if self.is_recording else "not recording"
                    status_color = (0, 255, 0) if self.is_recording else (0, 0, 255)  # 绿色=录制，红色=未录制

                    cv2.putText(display_frame,
                                f"Hand {idx + 1}: {hand_label} [{status_text}]",
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                status_color,
                                2)

                    y_offset += 30

                    # 绘制角度信息
                    for finger, finger_angles in smoothed_angles.items():
                        for joint, angle in finger_angles.items():
                            if angle is None:
                                continue

                            text = f"{finger}_{joint}: {angle:.1f}"
                            cv2.putText(display_frame,
                                        text,
                                        (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 255),  # BGR: 黄色
                                        1)
                            y_offset += 20

                # 显示帧索引
                cv2.putText(display_frame,
                            f"Frame: {current_frame_index}",
                            (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1)

                # ========== 生成压力热力图 ==========
                pressure_heatmap = self._generate_pressure_heatmap()

                # ========== 拼接视频和压力热力图 ==========
                combined_frame = self._combine_video_and_heatmap(display_frame, pressure_heatmap)

                cv2.imshow('MediaPipe + Pressure', combined_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                elif key == ord('r'):
                    # 按'r'开始/停止录制
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"显示错误: {e}")
                import traceback
                traceback.print_exc()

        # 清理 matplotlib 资源
        plt.close(self.pressure_fig)

    def _generate_pressure_heatmap(self):
        """生成压力热力图并返回为 numpy 数组 (BGR格式)"""
        # 获取最新压力数据
        with self.pressure_lock:
            if self.pressure_collector is not None and self.pressure_collector.latest_pressure_data is not None:
                pressure_data = self.pressure_collector.latest_pressure_data.copy()
            else:
                pressure_data = np.zeros(140)

        # 更新 scatter 数据
        self.pressure_scatter.set_array(pressure_data)

        # 渲染到 canvas
        canvas = FigureCanvas(self.pressure_fig)
        canvas.draw()

        # 转换为 numpy 数组
        width, height = self.pressure_fig.canvas.get_width_height()
        heatmap_rgb = np.frombuffer(self.pressure_fig.canvas.buffer_rgba(), dtype=np.uint8)
        heatmap_rgb = heatmap_rgb.reshape(height, width, 4)
        heatmap_rgb = heatmap_rgb[:, :, :3]  # 去掉 alpha 通道

        # RGB -> BGR (OpenCV 格式)
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

        return heatmap_bgr

    def _combine_video_and_heatmap(self, video_frame, heatmap):
        """将视频帧和压力热力图水平拼接"""
        video_h, video_w = video_frame.shape[:2]
        heatmap_h, heatmap_w = heatmap.shape[:2]

        # 调整热力图大小以匹配视频高度
        target_h = video_h
        scale = target_h / heatmap_h
        target_w = int(heatmap_w * scale)

        heatmap_resized = cv2.resize(heatmap, (target_w, target_h))

        # 水平拼接
        combined = np.hstack([video_frame, heatmap_resized])

        return combined

    def run(self):
        """
        运行手指角度检测和录制

        注意: 此方法会阻塞直到用户退出
        """
        # 检查OpenCV是否可用
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2_available = False
            print("错误: OpenCV未安装，无法使用摄像头")
            return

        # 如果要求使用线程但没有OpenCV，回退到非线程版本
        if self.use_threading and not cv2_available:
            print("警告: 线程模式需要OpenCV，已自动切换为非线程模式")
            self.use_threading = False

        # 检查模型文件是否存在
        if not os.path.exists(self.model_asset_path):
            print(f"警告: 模型文件 '{self.model_asset_path}' 不存在")
            print("MediaPipe可能会自动下载模型，但需要网络连接")
            print("或手动下载: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")

        # 初始化MediaPipe HandLandmarker
        print("初始化MediaPipe手部关键点检测器...")
        base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)


        if self.use_threading:
            self._run_threaded()
        else:
            print("Only can use threading")

    def _run_threaded(self):
        """运行线程版本"""
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.threads = []

        # 重置帧索引
        self.frame_index = 0

        # 启动线程
        self.threads.append(threading.Thread(target=self.capture_frames))
        self.threads.append(threading.Thread(target=self.inference_worker))
        self.threads.append(threading.Thread(target=self.display_worker))
        if self.whether_collecting_pressure_data:
            self.threads.append(threading.Thread(target=self.collect_pressure_data))

        for t in self.threads:
            t.daemon = True
            t.start()

        print("实时手指角度检测已启动")
        print("按 'r' 开始/停止录制")
        print("按 'q' 退出")

        # 等待停止事件
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        # 清理
        cv2.destroyAllWindows()

        # 停止录制（如果正在录制）
        if self.is_recording:
            self.stop_recording()

        print("检测已停止")


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 创建收集器实例
    collector = FingerAngleAndPressureSensorCollector(
        record_dir="record",
        camera_id=0,
        use_threading=True,
        num_hands=2,
        model_asset_path='hand_landmarker.task',
        video_codec='mp4v',
        video_fps=30.0
    )

    # 运行收集器
    collector.run()