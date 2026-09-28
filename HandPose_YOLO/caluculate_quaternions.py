from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

# --------------------------
# 1. 加载YOLO姿态估计模型
# --------------------------
model = YOLO("yolo26n-pose.pt")  # 轻量版，速度快；也可用yolov8n-pose.pt


# --------------------------
# 2. 定义核心函数：计算手臂关节旋转四元数
# --------------------------
def calculate_arm_quaternion(keypoints, side="right"):
    """
    计算单只手臂（肩-肘-腕）的旋转四元数
    参数：
        keypoints: YOLO输出的关键点数组，形状为(17, 3)，每一行是(x, y, conf)
        side: "right"（右臂）或 "left"（左臂）
    返回：
        quaternion: 旋转四元数 (w, x, y, z)；None表示关键点无效
    """
    # 1. 选择对应手臂的关键点索引
    if side == "right":
        shoulder_idx = 5
        elbow_idx = 7
        wrist_idx = 9
    else:
        shoulder_idx = 6
        elbow_idx = 8
        wrist_idx = 10

    # 2. 提取关键点坐标并过滤低置信度（conf<0.5）的点
    shoulder = keypoints[shoulder_idx]  # (x, y, conf)
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]

    # 检查关键点是否有效（置信度>0.5）
    if shoulder[2] < 0.5 or elbow[2] < 0.5 or wrist[2] < 0.5:
        print(f"{side}手臂关键点置信度不足，跳过计算")
        return None

    # 3. 转换为2D向量（仅用x,y，忽略z）
    # 以肩膀为原点，构建局部坐标系
    shoulder_xy = np.array([shoulder[0], shoulder[1]])
    elbow_xy = np.array([elbow[0], elbow[1]])
    wrist_xy = np.array([wrist[0], wrist[1]])

    # 4. 计算方向向量
    vec1 = elbow_xy - shoulder_xy  # 肩→肘向量
    vec2 = wrist_xy - elbow_xy  # 肘→腕向量

    # 5. 扩展为3D向量（假设z=0，平面旋转）
    vec1_3d = np.array([vec1[0], vec1[1], 0.0])
    vec2_3d = np.array([vec2[0], vec2[1], 0.0])

    # 6. 计算旋转矩阵：拟合从参考向量到当前向量的旋转
    # 参考向量（默认手臂向前）：肩→肘沿x轴，肘→腕沿x轴
    '''
    若想以 “手臂向前（x 轴）” 为参考，可设ref_vec1 = (1,0,0)
    若想以 “手臂向上（y 轴）” 为参考，可设ref_vec1 = (0,1,0)
    若想以 “手臂向外（z 轴）” 为参考，可设ref_vec1 = (0,0,1)
    '''
    ref_vec1 = np.array([0.0, 1.0, 0.0])
    ref_vec2 = np.array([0.0, 1.0, 0.0])

    # 归一化向量
    vec1_3d = vec1_3d / np.linalg.norm(vec1_3d)
    vec2_3d = vec2_3d / np.linalg.norm(vec2_3d)
    ref_vec1 = ref_vec1 / np.linalg.norm(ref_vec1)
    ref_vec2 = ref_vec2 / np.linalg.norm(ref_vec2)

    # 构建旋转矩阵（SVD方法，避免万向锁）
    v = np.cross(ref_vec1, vec1_3d)
    c = np.dot(ref_vec1, vec1_3d)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_mat = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2 + 1e-8))

    # 7. 旋转矩阵转四元数（scipy内置方法，避免手动计算出错）
    rotation = R.from_matrix(rot_mat)
    quaternion = rotation.as_quat()  # 输出格式：[x, y, z, w]
    # 调整为(w, x, y, z)格式（更通用的四元数表示）
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

    return quaternion


# --------------------------
# 3. 推理并计算四元数
# --------------------------
def process_arm_pose(image_path):
    # 推理（关闭日志输出）
    results = model(image_path, verbose=False)
    # 提取第一个结果的关键点（shape: (17, 3)）
    keypoints = results[0].keypoints.data.cpu().numpy()[0]  # 转numpy并取第一个人体

    # 计算左右手臂的旋转四元数
    right_quat = calculate_arm_quaternion(keypoints, side="right")
    left_quat = calculate_arm_quaternion(keypoints, side="left")

    # 输出结果
    print("=== 手臂旋转四元数（w, x, y, z）===")
    print(f"右臂：{right_quat}")
    print(f"左臂：{left_quat}")

    # 可选：可视化关键点（方便验证）
    annotated_img = results[0].plot()
    cv2.imshow("Arm Keypoints", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------
# 4. 运行测试
# --------------------------
if __name__ == "__main__":
    # 替换为你的图片路径（建议用清晰的手臂正面/侧面照）
    image_path = "arm_test.jpg"
    process_arm_pose(image_path)