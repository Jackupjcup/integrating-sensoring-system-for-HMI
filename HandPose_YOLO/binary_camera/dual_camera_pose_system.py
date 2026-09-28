"""
双摄像头人体姿态3D重建与四元数计算系统
基于YOLO26n-Pose模型，通过三角测量获取3D关节坐标，计算肢体段四元数姿态
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import time
from collections import deque


class DualCameraPoseSystem:
    """双摄像头人体姿态估计系统"""

    # COCO关键点索引（YOLO-Pose使用COCO格式）
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }

    # 手臂相关关键点
    ARM_KEYPOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]

    def __init__(self, calibration_file="calibration.json", model_path="yolo26n-pose.pt"):
        """
        初始化系统

        参数:
            calibration_file: 标定参数文件路径
            model_path: YOLO模型路径
        """
        # 加载标定参数
        self.load_calibration(calibration_file)

        # 加载YOLO模型
        print(f"加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)

        # 计算投影矩阵
        self.P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.K2, np.hstack((self.R, self.T.reshape(3, 1))))

        # 历史数据缓存（用于平滑）
        self.history_3d = {}
        self.history_quaternions = {}
        self.history_length = 5  # 缓存长度

        # 性能统计
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)

        print(f"系统初始化完成，基线距离: {np.linalg.norm(self.T):.3f}米")

    def load_calibration(self, calibration_file):
        """加载标定参数"""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)

            # 摄像头内参
            self.K1 = np.array(calib_data["camera1"]["K"])
            self.D1 = np.array(calib_data["camera1"]["D"])
            self.K2 = np.array(calib_data["camera2"]["K"])
            self.D2 = np.array(calib_data["camera2"]["D"])

            # 外参（从camera1到camera2）
            self.R = np.array(calib_data["extrinsic"]["R"])
            self.T = np.array(calib_data["extrinsic"]["T"])

            # 标定误差
            self.calib_error = calib_data["extrinsic"]["error"]

            print(f"标定参数已加载，重投影误差: {self.calib_error:.4f}")

        except Exception as e:
            print(f"加载标定参数失败: {e}")
            print("使用默认参数（需要先运行标定）")
            # 默认参数（仅用于测试）
            self.K1 = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            self.D1 = np.zeros((1, 5), dtype=np.float32)
            self.K2 = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            self.D2 = np.zeros((1, 5), dtype=np.float32)
            self.R = np.eye(3, dtype=np.float32)
            self.T = np.array([0.3, 0, 0], dtype=np.float32).reshape(3, 1)

    def detect_keypoints_2d(self, frame1, frame2, confidence_threshold=0.3):
        """
        在两个摄像头帧中检测2D关键点

        参数:
            frame1: 摄像头1图像
            frame2: 摄像头2图像
            confidence_threshold: 关键点置信度阈值

        返回:
            kpts1, kpts2: (17, 3)数组，每行包含(x, y, confidence)，失败返回None
        """
        start_time = time.time()

        # 运行YOLO推理
        results1 = self.model(frame1, verbose=False)[0]
        results2 = self.model(frame2, verbose=False)[0]

        # 提取关键点
        kpts1 = results1.keypoints.data.cpu().numpy() if results1.keypoints is not None else None
        kpts2 = results2.keypoints.data.cpu().numpy() if results2.keypoints is not None else None

        # 选择置信度最高的人体
        def select_best_person(kpts):
            if kpts is None or len(kpts) == 0:
                return None

            # 计算平均置信度
            confidences = kpts[:, :, 2].mean(axis=1)
            best_idx = np.argmax(confidences)

            if confidences[best_idx] < confidence_threshold:
                return None

            return kpts[best_idx]  # (17, 3)

        kpts1_best = select_best_person(kpts1)
        kpts2_best = select_best_person(kpts2)

        # 性能统计
        detect_time = time.time() - start_time
        self.processing_times.append(detect_time)

        return kpts1_best, kpts2_best

    def triangulate_keypoints(self, kpts1, kpts2, min_confidence=0.3):
        """
        三角测量计算3D坐标

        参数:
            kpts1, kpts2: (17, 3)关键点数组
            min_confidence: 最小置信度阈值

        返回:
            points_3d: 字典，键为关键点名称，值为3D坐标 (x, y, z)
        """
        if kpts1 is None or kpts2 is None:
            return None

        points_1 = []
        points_2 = []
        valid_indices = []
        valid_names = []

        # 收集有效的关键点对
        for name, idx in self.KEYPOINT_INDICES.items():
            # 检查置信度
            if (kpts1[idx, 2] > min_confidence and kpts2[idx, 2] > min_confidence and
                kpts1[idx, 0] > 0 and kpts1[idx, 1] > 0 and
                kpts2[idx, 0] > 0 and kpts2[idx, 1] > 0):

                points_1.append(kpts1[idx, :2])  # (x, y)
                points_2.append(kpts2[idx, :2])
                valid_indices.append(idx)
                valid_names.append(name)

        if len(points_1) < 4:  # 至少需要4个点进行稳定的三角测量
            return None

        # 转换为numpy数组
        points_1 = np.array(points_1).T  # (2, n)
        points_2 = np.array(points_2).T  # (2, n)

        # 三角测量
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points_1, points_2)

        # 转换为3D坐标 (齐次坐标→笛卡尔坐标)
        points_3d = points_4d[:3] / points_4d[3]

        # 构建结果字典
        points_3d_dict = {}
        for i, name in enumerate(valid_names):
            points_3d_dict[name] = points_3d[:, i]

        return points_3d_dict

    def calculate_limb_quaternions(self, points_3d):
        """
        计算肢体段四元数姿态

        参数:
            points_3d: 3D关键点坐标字典

        返回:
            quaternions: 字典，包含各肢体段的四元数 (w, x, y, z)
        """
        quaternions = {}

        # 计算右臂姿态
        if all(k in points_3d for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            shoulder = points_3d["right_shoulder"]
            elbow = points_3d["right_elbow"]
            wrist = points_3d["right_wrist"]

            # 计算向量
            upper_arm_vec = elbow - shoulder  # 上臂向量
            forearm_vec = wrist - elbow       # 前臂向量

            # 计算上臂四元数（相对于世界坐标系）
            upper_arm_quat = self._vector_to_quaternion(upper_arm_vec, reference=np.array([1, 0, 0]))

            # 计算肘关节四元数（上臂到前臂的相对旋转）
            elbow_quat = self._relative_rotation(upper_arm_vec, forearm_vec)

            quaternions["right_upper_arm"] = upper_arm_quat
            quaternions["right_elbow"] = elbow_quat

            # 计算手腕方向（假设手腕方向与前臂方向一致）
            wrist_quat = self._vector_to_quaternion(forearm_vec, reference=np.array([1, 0, 0]))
            quaternions["right_wrist"] = wrist_quat

        # 计算左臂姿态
        if all(k in points_3d for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            shoulder = points_3d["left_shoulder"]
            elbow = points_3d["left_elbow"]
            wrist = points_3d["left_wrist"]

            upper_arm_vec = elbow - shoulder
            forearm_vec = wrist - elbow

            upper_arm_quat = self._vector_to_quaternion(upper_arm_vec, reference=np.array([1, 0, 0]))
            elbow_quat = self._relative_rotation(upper_arm_vec, forearm_vec)

            quaternions["left_upper_arm"] = upper_arm_quat
            quaternions["left_elbow"] = elbow_quat

            wrist_quat = self._vector_to_quaternion(forearm_vec, reference=np.array([1, 0, 0]))
            quaternions["left_wrist"] = wrist_quat

        # 计算脊柱方向（肩膀中心到髋部中心）
        if all(k in points_3d for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            shoulder_center = (points_3d["left_shoulder"] + points_3d["right_shoulder"]) / 2
            hip_center = (points_3d["left_hip"] + points_3d["right_hip"]) / 2
            spine_vec = hip_center - shoulder_center

            if np.linalg.norm(spine_vec) > 0.1:  # 避免零向量
                spine_quat = self._vector_to_quaternion(spine_vec, reference=np.array([0, 0, 1]))
                quaternions["spine"] = spine_quat

        return quaternions

    def _vector_to_quaternion(self, vector, reference):
        """
        将向量转换为四元数（表示从参考向量到目标向量的旋转）

        参数:
            vector: 目标向量
            reference: 参考向量

        返回:
            quaternion: 四元数 (w, x, y, z)
        """
        # 归一化向量
        v1 = reference / (np.linalg.norm(reference) + 1e-8)
        v2 = vector / (np.linalg.norm(vector) + 1e-8)

        # 计算旋转轴和角度
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # 向量平行或反平行
            if np.dot(v1, v2) > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])  # 无旋转
            else:
                # 180度旋转，需要任意垂直轴
                axis = np.array([0, 1, 0]) if abs(v1[0]) < 0.9 else np.array([1, 0, 0])
                axis = axis / np.linalg.norm(axis)
                angle = np.pi
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        # 四元数: (w, x, y, z)
        w = np.cos(angle / 2)
        x, y, z = axis * np.sin(angle / 2)

        return np.array([w, x, y, z])

    def _relative_rotation(self, vec1, vec2):
        """计算从vec1到vec2的相对旋转"""
        # 归一化
        v1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        v2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # 直接计算旋转四元数
        return self._vector_to_quaternion(v2, v1)

    def apply_temporal_smoothing(self, points_3d, quaternions, smoothing_factor=0.3):
        """
        应用时间平滑

        参数:
            points_3d: 当前帧的3D点
            quaternions: 当前帧的四元数
            smoothing_factor: 平滑系数 (0-1)，越小越平滑

        返回:
            smoothed_points_3d, smoothed_quaternions
        """
        smoothed_points_3d = {}
        smoothed_quaternions = {}

        # 平滑3D点
        for name, point in points_3d.items():
            if name not in self.history_3d:
                self.history_3d[name] = deque([point.copy()], maxlen=self.history_length)
            else:
                self.history_3d[name].append(point.copy())

            # 指数加权平均
            if len(self.history_3d[name]) == 1:
                smoothed_points_3d[name] = point
            else:
                # 计算加权平均
                weights = np.exp(-smoothing_factor * np.arange(len(self.history_3d[name])))
                weights = weights / weights.sum()

                points_array = np.array(list(self.history_3d[name]))
                smoothed_point = np.sum(points_array * weights[:, np.newaxis], axis=0)
                smoothed_points_3d[name] = smoothed_point

        # 平滑四元数（使用球面线性插值Slerp的简化版本）
        for name, quat in quaternions.items():
            if name not in self.history_quaternions:
                self.history_quaternions[name] = deque([quat.copy()], maxlen=self.history_length)
            else:
                self.history_quaternions[name].append(quat.copy())

            if len(self.history_quaternions[name]) == 1:
                smoothed_quaternions[name] = quat
            else:
                # 简单的四元数平均（对于小角度变化足够好）
                quats_array = np.array(list(self.history_quaternions[name]))
                # 确保所有四元数在同一个半球
                if np.sum(quats_array[:, 0]) < 0:
                    quats_array = -quats_array

                avg_quat = np.mean(quats_array, axis=0)
                avg_quat = avg_quat / np.linalg.norm(avg_quat)
                smoothed_quaternions[name] = avg_quat

        return smoothed_points_3d, smoothed_quaternions

    def process_frame(self, frame1, frame2, apply_smoothing=True):
        """
        处理单帧图像

        参数:
            frame1: 摄像头1图像
            frame2: 摄像头2图像
            apply_smoothing: 是否应用时间平滑

        返回:
            points_3d: 3D关键点坐标字典
            quaternions: 四元数姿态字典
            visualization_data: 可视化数据
        """
        start_time = time.time()

        # 1. 2D关键点检测
        kpts1, kpts2 = self.detect_keypoints_2d(frame1, frame2)

        if kpts1 is None or kpts2 is None:
            return None, None, None

        # 2. 三角测量得到3D坐标
        points_3d_raw = self.triangulate_keypoints(kpts1, kpts2)

        if points_3d_raw is None:
            return None, None, None

        # 3. 计算四元数
        quaternions_raw = self.calculate_limb_quaternions(points_3d_raw)

        # 4. 时间平滑
        if apply_smoothing:
            points_3d, quaternions = self.apply_temporal_smoothing(points_3d_raw, quaternions_raw)
        else:
            points_3d, quaternions = points_3d_raw, quaternions_raw

        # 5. 准备可视化数据
        visualization_data = {
            "kpts1": kpts1,
            "kpts2": kpts2,
            "frame1": frame1.copy(),
            "frame2": frame2.copy(),
            "processing_time": time.time() - start_time
        }

        self.frame_count += 1

        return points_3d, quaternions, visualization_data

    def draw_visualization(self, visualization_data, points_3d=None, quaternions=None):
        """
        绘制可视化结果

        参数:
            visualization_data: 从process_frame返回的可视化数据
            points_3d: 3D关键点坐标（可选）
            quaternions: 四元数姿态（可选）

        返回:
            vis_frame1, vis_frame2: 可视化后的图像
        """
        if visualization_data is None:
            return None, None

        kpts1 = visualization_data["kpts1"]
        kpts2 = visualization_data["kpts2"]
        frame1 = visualization_data["frame1"]
        frame2 = visualization_data["frame2"]

        # 创建副本
        vis_frame1 = frame1.copy()
        vis_frame2 = frame2.copy()

        # 绘制2D关键点
        def draw_keypoints(frame, kpts, color=(0, 255, 0)):
            if kpts is None:
                return

            # 绘制点
            for i, (x, y, conf) in enumerate(kpts):
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                    cv2.putText(frame, str(i), (int(x)+5, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 绘制骨架连线
            skeleton = [
                (5, 7), (7, 9),  # 左臂
                (6, 8), (8, 10),  # 右臂
                (5, 6),  # 肩膀
                (5, 11), (6, 12),  # 身体
                (11, 13), (13, 15),  # 左腿
                (12, 14), (14, 16),  # 右腿
            ]

            for idx1, idx2 in skeleton:
                if kpts[idx1, 2] > 0.3 and kpts[idx2, 2] > 0.3:
                    x1, y1 = int(kpts[idx1, 0]), int(kpts[idx1, 1])
                    x2, y2 = int(kpts[idx2, 0]), int(kpts[idx2, 1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        draw_keypoints(vis_frame1, kpts1)
        draw_keypoints(vis_frame2, kpts2)

        # 添加信息文本
        processing_time = visualization_data.get("processing_time", 0)
        fps = 1.0 / processing_time if processing_time > 0 else 0

        info_text = f"FPS: {fps:.1f} | Points: {len(points_3d) if points_3d else 0}"
        cv2.putText(vis_frame1, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame2, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 如果提供了3D点和四元数，显示在图像上
        if points_3d is not None and quaternions is not None:
            # 显示关键3D点
            y_offset = 60
            for name in ["right_shoulder", "right_elbow", "right_wrist"]:
                if name in points_3d:
                    coord = points_3d[name]
                    text = f"{name}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})"
                    cv2.putText(vis_frame1, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20

            # 显示四元数
            y_offset = 60
            for name in ["right_upper_arm", "right_elbow"]:
                if name in quaternions:
                    q = quaternions[name]
                    text = f"{name}: ({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f})"
                    cv2.putText(vis_frame2, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20

        return vis_frame1, vis_frame2

    def print_results(self, points_3d, quaternions, frame_num=None):
        """打印结果到控制台"""
        if frame_num is None:
            frame_num = self.frame_count

        print(f"\n=== 帧 {frame_num} 结果 ===")

        if points_3d:
            print("3D关键点坐标:")
            for name in self.ARM_KEYPOINTS:
                if name in points_3d:
                    coord = points_3d[name]
                    print(f"  {name}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")

        if quaternions:
            print("四元数姿态 (w, x, y, z):")
            for name in ["right_upper_arm", "right_elbow", "left_upper_arm", "left_elbow"]:
                if name in quaternions:
                    q = quaternions[name]
                    print(f"  {name}: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")

        # 显示性能信息
        if len(self.processing_times) > 0:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"性能: {avg_time*1000:.1f}ms/帧 ({fps:.1f} FPS)")

    def get_performance_stats(self):
        """获取性能统计"""
        if len(self.processing_times) == 0:
            return {"fps": 0, "avg_time_ms": 0}

        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "fps": fps,
            "avg_time_ms": avg_time * 1000,
            "frames_processed": self.frame_count,
            "calibration_error": self.calib_error
        }


def test_system_with_images(image1_path, image2_path, calibration_file="calibration.json"):
    """使用静态图像测试系统"""
    print("使用静态图像测试系统...")

    # 加载图像
    frame1 = cv2.imread(image1_path)
    frame2 = cv2.imread(image2_path)

    if frame1 is None or frame2 is None:
        print("无法加载图像")
        return

    # 初始化系统
    system = DualCameraPoseSystem(calibration_file)

    # 处理帧
    points_3d, quaternions, vis_data = system.process_frame(frame1, frame2, apply_smoothing=False)

    # 显示结果
    if points_3d is not None:
        system.print_results(points_3d, quaternions)

        # 可视化
        vis_frame1, vis_frame2 = system.draw_visualization(vis_data, points_3d, quaternions)

        cv2.imshow("Camera 1", vis_frame1)
        cv2.imshow("Camera 2", vis_frame2)

        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能检测到有效的人体姿态")


if __name__ == "__main__":
    # 测试代码
    test_system_with_images("test_cam1.jpg", "test_cam2.jpg")