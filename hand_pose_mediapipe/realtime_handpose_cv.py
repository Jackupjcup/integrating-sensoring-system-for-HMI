#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from AllData_collect_2 import calculate_finger_angles_mediapipe, FingerAngleAndPressureSensorCollector

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# 手部关键点连接定义（MediaPipe标准，21个关键点）
# 作为mp_hands.HAND_CONNECTIONS的后备方案
HAND_CONNECTIONS_LIST = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
    (5, 9), (9, 13), (13, 17)                 # 手掌连接
]

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# ============================================================================
# 单张图片手指角度计算函数
# ============================================================================

def calculate_finger_angles_from_image(image_path, num_hands=2):
    """
    计算单张图片中手指关节角度

    参数:
        image_path: 图片文件路径
        num_hands: 检测的最大手部数量

    返回:
        angles_list: 列表，包含每只手的手指角度字典
    """
    # 初始化检测器（使用默认模型路径）
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 加载图片
    image = mp.Image.create_from_file(image_path)

    # 检测手部关键点
    detection_result = detector.detect(image)
    processed_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite("mediapipe_image.jpg", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

    # 获取手部关键点列表
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    angles_list = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # 计算手指角度
        angles = calculate_finger_angles_mediapipe(hand_landmarks)

        angles_list.append({
            'hand_index': idx,
            'handedness': handedness[0].category_name,
            'angles': angles
        })

        # 打印结果
        print(f"\n{'='*50}")
        print(f"手部 {idx+1} ({handedness[0].category_name}) 手指角度:")
        for finger, finger_angles in angles.items():
            print(f"  {finger}:")
            for joint, angle in finger_angles.items():
                if angle is not None:
                    print(f"    {joint}: {angle:.1f}°")

    return angles_list


def replace_white_with_skin(frame, skin_color=(190, 170, 150)):
    """
    将帧中的白色区域替换为肉色
    :param frame: 输入帧（BGR格式，cv2读取的默认格式）
    :param skin_color: 肉色BGR值，默认(190, 170, 150)
    :return: 替换后的帧
    """
    # 1. 转换为HSV颜色空间（便于颜色分割）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 定义白色的HSV阈值（可根据实际场景微调）
    # 低阈值：H=0, S=0, V=220；高阈值：H=180, S=30, V=255
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 30, 255])

    # 3. 生成白色区域的掩码（255=白色区域，0=非白色区域）
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 4. 可选：对掩码进行形态学操作（去除噪点，让替换更平滑）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：填充小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：去除小噪点

    # 5. 创建肉色背景图（和原帧尺寸一致）
    skin_frame = np.full(frame.shape, skin_color, dtype=np.uint8)

    # 6. 合并：掩码区域显示肉色，非掩码区域显示原帧
    # cv2.bitwise_and：只保留掩码为255的区域
    result = np.where(mask[:, :, np.newaxis] == 255, skin_frame, frame)

    return result, mask  # 返回结果帧+掩码（便于调试）


if __name__ == "__main__":
    # 创建收集器实例
    collector = FingerAngleAndPressureSensorCollector(
        record_dir="record",
        camera_id=0,
        use_threading=True,
        num_hands=2,
        model_asset_path='hand_landmarker.task',
        video_codec='mp4v',
        video_fps=30.0,
        whether_collecting_pressure_data=False
    )

    # 运行收集器
    collector.run()

