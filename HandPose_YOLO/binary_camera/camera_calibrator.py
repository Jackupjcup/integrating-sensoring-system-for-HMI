"""
双摄像头标定工具
支持棋盘格和Charuco板标定，计算内参、畸变系数和摄像头间外参
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器，支持numpy数组"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CameraCalibrator:
    """摄像头标定类"""

    def __init__(self, pattern_type="checkerboard", pattern_size=(9, 6), square_size=0.025):
        """
        初始化标定器

        参数:
            pattern_type: 标定板类型 - "checkerboard"（棋盘格）或 "charuco"（Charuco板）
            pattern_size: 内角点数量 (列, 行)
            square_size: 方格尺寸（米）
        """
        self.pattern_type = pattern_type
        self.pattern_size = pattern_size  # (width, height)
        self.square_size = square_size

        # 标定结果
        self.camera1_params = {"K": None, "D": None, "error": None}
        self.camera2_params = {"K": None, "D": None, "error": None}
        self.extrinsic_params = {"R": None, "T": None, "error": None}

        # 标定板准备
        if pattern_type == "charuco":
            self._init_charuco_board()

    def _init_charuco_board(self):
        """初始化Charuco板"""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.charuco_board = cv2.aruco.CharucoBoard(
            self.pattern_size, self.square_size, self.square_size * 0.8, self.aruco_dict
        )

    def collect_calibration_images(self, cam1_idx=0, cam2_idx=1, num_images=20, output_dir="calibration_images"):
        """
        采集标定图像

        参数:
            cam1_idx: 摄像头1索引
            cam2_idx: 摄像头2索引
            num_images: 需要采集的图像对数
            output_dir: 保存图像的目录
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 打开摄像头
        cap1 = cv2.VideoCapture(cam1_idx)
        cap2 = cv2.VideoCapture(cam2_idx)

        # 设置相同参数
        for cap in [cap1, cap2]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        print("=" * 50)
        print("标定图像采集")
        print("=" * 50)
        print(f"需要采集 {num_images} 对图像")
        print("提示:")
        print("1. 将标定板置于两个摄像头共同视野中")
        print("2. 移动标定板到不同位置和角度")
        print("3. 确保标定板在两个画面中都清晰可见")
        print("4. 按 'c' 键捕获图像，按 'q' 键退出")
        print("=" * 50)

        collected = 0
        while collected < num_images:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("摄像头读取失败")
                break

            # 检测标定板（仅用于预览）
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if self.pattern_type == "checkerboard":
                ret1_detect, corners1 = cv2.findChessboardCorners(gray1, self.pattern_size, None)
                ret2_detect, corners2 = cv2.findChessboardCorners(gray2, self.pattern_size, None)
            else:  # charuco
                ret1_detect, corners1, ids1 = cv2.aruco.detectMarkers(gray1, self.aruco_dict)
                ret2_detect, corners2, ids2 = cv2.aruco.detectMarkers(gray2, self.aruco_dict)

            # 绘制检测结果
            display1 = frame1.copy()
            display2 = frame2.copy()

            if ret1_detect:
                if self.pattern_type == "checkerboard":
                    cv2.drawChessboardCorners(display1, self.pattern_size, corners1, ret1_detect)
                else:
                    cv2.aruco.drawDetectedMarkers(display1, corners1, ids1)
                cv2.putText(display1, "Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ret2_detect:
                if self.pattern_type == "checkerboard":
                    cv2.drawChessboardCorners(display2, self.pattern_size, corners2, ret2_detect)
                else:
                    cv2.aruco.drawDetectedMarkers(display2, corners2, ids2)
                cv2.putText(display2, "Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示图像
            cv2.putText(display1, f"Collected: {collected}/{num_images}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display2, f"Collected: {collected}/{num_images}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera 1 - Press 'c' to capture, 'q' to quit", display1)
            cv2.imshow("Camera 2 - Press 'c' to capture, 'q' to quit", display2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # 只在两个摄像头都检测到标定板时保存
                if ret1_detect and ret2_detect:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img1_path = os.path.join(output_dir, f"cam1_{timestamp}.jpg")
                    img2_path = os.path.join(output_dir, f"cam2_{timestamp}.jpg")

                    cv2.imwrite(img1_path, frame1)
                    cv2.imwrite(img2_path, frame2)

                    collected += 1
                    print(f"已采集图像对: {collected}/{num_images}")
                else:
                    print("警告: 标定板未在两个画面中同时检测到")

            elif key == ord('q'):
                print("用户中断采集")
                break

        # 释放资源
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        print(f"采集完成，共获得 {collected} 对图像")
        return collected

    def calibrate_single_camera(self, image_paths, camera_name="camera1"):
        """
        标定单个摄像头内参

        参数:
            image_paths: 图像路径列表
            camera_name: 摄像头名称（用于输出信息）

        返回:
            K: 相机内参矩阵
            D: 畸变系数
            error: 重投影误差
        """
        print(f"开始标定 {camera_name}...")

        obj_points = []  # 3D世界坐标
        img_points = []  # 2D图像坐标

        # 生成标定板角点的3D坐标
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        image_size = None

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)

            if self.pattern_type == "checkerboard":
                # 查找棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

                if ret:
                    # 亚像素优化
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    obj_points.append(objp)
                    img_points.append(corners_refined)

                    if i % 5 == 0:
                        print(f" 处理图像 {i+1}/{len(image_paths)}: 找到角点")
                else:
                    print(f" 处理图像 {i+1}/{len(image_paths)}: 未找到角点")

            else:  # charuco
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

                if len(corners) > 0:
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, self.charuco_board
                    )

                    if ret > 0:
                        obj_points.append(objp)
                        img_points.append(charuco_corners)

                        if i % 5 == 0:
                            print(f" 处理图像 {i+1}/{len(image_paths)}: 找到 {ret} 个Charuco角点")
                    else:
                        print(f" 处理图像 {i+1}/{len(image_paths)}: Charuco角点不足")
                else:
                    print(f" 处理图像 {i+1}/{len(image_paths)}: 未找到ArUco标记")

        if len(obj_points) < 10:
            print(f"错误: {camera_name} 有效图像不足（需要至少10张，当前{len(obj_points)}张）")
            return None, None, None

        print(f"使用 {len(obj_points)} 张图像进行标定...")

        # 摄像头标定
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        # 计算重投影误差
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(obj_points)

        print(f"{camera_name} 标定完成:")
        print(f"  重投影误差: {mean_error:.4f} 像素")
        print(f"  内参矩阵 K:\n{K}")
        print(f"  畸变系数 D: {D.ravel()}")

        if camera_name == "camera1":
            self.camera1_params = {"K": K, "D": D, "error": mean_error}
        else:
            self.camera2_params = {"K": K, "D": D, "error": mean_error}

        return K, D, mean_error

    def calibrate_stereo(self, paired_images):
        """
        标定双目摄像头外参

        参数:
            paired_images: 成对图像路径列表 [(img1_path, img2_path), ...]

        返回:
            R: 旋转矩阵（从camera1到camera2）
            T: 平移向量（从camera1到camera2）
            error: 立体标定误差
        """
        print("开始双目标定...")

        obj_points = []  # 3D世界坐标
        img_points1 = []  # camera1图像坐标
        img_points2 = []  # camera2图像坐标

        # 生成标定板角点的3D坐标
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        image_size = None

        for i, (img1_path, img2_path) in enumerate(paired_images):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"警告: 无法读取图像对 {img1_path}, {img2_path}")
                continue

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            if image_size is None:
                image_size = gray1.shape[::-1]

            success_pair = False

            if self.pattern_type == "checkerboard":
                # 查找棋盘格角点
                ret1, corners1 = cv2.findChessboardCorners(gray1, self.pattern_size, None)
                ret2, corners2 = cv2.findChessboardCorners(gray2, self.pattern_size, None)

                if ret1 and ret2:
                    # 亚像素优化
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners1_refined = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2_refined = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                    obj_points.append(objp)
                    img_points1.append(corners1_refined)
                    img_points2.append(corners2_refined)
                    success_pair = True

            else:  # charuco
                corners1, ids1, _ = cv2.aruco.detectMarkers(gray1, self.aruco_dict)
                corners2, ids2, _ = cv2.aruco.detectMarkers(gray2, self.aruco_dict)

                if len(corners1) > 0 and len(corners2) > 0:
                    ret1, charuco_corners1, charuco_ids1 = cv2.aruco.interpolateCornersCharuco(
                        corners1, ids1, gray1, self.charuco_board
                    )
                    ret2, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(
                        corners2, ids2, gray2, self.charuco_board
                    )

                    if ret1 > 0 and ret2 > 0:
                        obj_points.append(objp)
                        img_points1.append(charuco_corners1)
                        img_points2.append(charuco_corners2)
                        success_pair = True

            if success_pair and i % 5 == 0:
                print(f" 处理图像对 {i+1}/{len(paired_images)}: 成功")
            elif not success_pair:
                print(f" 处理图像对 {i+1}/{len(paired_images)}: 失败")

        if len(obj_points) < 10:
            print(f"错误: 有效图像对不足（需要至少10对，当前{len(obj_points)}对）")
            return None, None, None

        print(f"使用 {len(obj_points)} 对图像进行立体标定...")

        # 立体标定
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points1, img_points2,
            self.camera1_params["K"], self.camera1_params["D"],
            self.camera2_params["K"], self.camera2_params["D"],
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        self.extrinsic_params = {"R": R, "T": T, "error": ret}

        print("双目标定完成:")
        print(f"  标定误差: {ret:.4f}")
        print(f"  旋转矩阵 R:\n{R}")
        print(f"  平移向量 T: {T.ravel()}")
        print(f"  基线距离: {np.linalg.norm(T):.3f} 米")

        return R, T, ret

    def save_calibration(self, filepath="calibration.json"):
        """保存标定参数到JSON文件"""
        if (self.camera1_params["K"] is None or self.camera2_params["K"] is None or
            self.extrinsic_params["R"] is None):
            print("错误: 标定参数不完整，无法保存")
            return False

        calib_data = {
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_type": self.pattern_type,
            "pattern_size": self.pattern_size,
            "square_size": self.square_size,
            "camera1": self.camera1_params,
            "camera2": self.camera2_params,
            "extrinsic": self.extrinsic_params,
            "projection_matrices": {
                "P1": self._calculate_projection_matrix(1).tolist(),
                "P2": self._calculate_projection_matrix(2).tolist()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(calib_data, f, cls=NumpyEncoder, indent=2)

        print(f"标定参数已保存到 {filepath}")
        return True

    def _calculate_projection_matrix(self, camera_idx):
        """计算投影矩阵"""
        if camera_idx == 1:
            K = self.camera1_params["K"]
            return np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        else:
            K = self.camera2_params["K"]
            R = self.extrinsic_params["R"]
            T = self.extrinsic_params["T"]
            return np.dot(K, np.hstack((R, T.reshape(3, 1))))

    def load_calibration(self, filepath="calibration.json"):
        """从JSON文件加载标定参数"""
        try:
            with open(filepath, 'r') as f:
                calib_data = json.load(f)

            self.pattern_type = calib_data.get("pattern_type", "checkerboard")
            self.pattern_size = tuple(calib_data.get("pattern_size", (9, 6)))
            self.square_size = calib_data.get("square_size", 0.025)

            self.camera1_params = {
                "K": np.array(calib_data["camera1"]["K"]),
                "D": np.array(calib_data["camera1"]["D"]),
                "error": calib_data["camera1"]["error"]
            }

            self.camera2_params = {
                "K": np.array(calib_data["camera2"]["K"]),
                "D": np.array(calib_data["camera2"]["D"]),
                "error": calib_data["camera2"]["error"]
            }

            self.extrinsic_params = {
                "R": np.array(calib_data["extrinsic"]["R"]),
                "T": np.array(calib_data["extrinsic"]["T"]),
                "error": calib_data["extrinsic"]["error"]
            }

            print(f"标定参数已从 {filepath} 加载")
            return True

        except Exception as e:
            print(f"加载标定参数失败: {e}")
            return False


def run_calibration_pipeline():
    """运行完整的标定流程"""
    print("=" * 60)
    print("双摄像头标定流程")
    print("=" * 60)

    # 1. 初始化标定器
    calibrator = CameraCalibrator(
        pattern_type="checkerboard",  # 或 "charuco"
        pattern_size=(9, 6),
        square_size=0.025
    )

    # 2. 采集标定图像
    print("\n步骤1: 采集标定图像")
    print("-" * 40)
    collected = calibrator.collect_calibration_images(
        cam1_idx=0,
        cam2_idx=1,
        num_images=20,
        output_dir="calibration_images"
    )

    if collected < 10:
        print("采集的图像不足，标定可能不准确")

    # 3. 加载图像路径
    print("\n步骤2: 加载图像进行标定")
    print("-" * 40)

    import glob
    cam1_images = sorted(glob.glob("calibration_images/cam1_*.jpg"))
    cam2_images = sorted(glob.glob("calibration_images/cam2_*.jpg"))

    if len(cam1_images) != len(cam2_images):
        print("警告: 两个摄像头的图像数量不匹配")
        # 使用较短的长度
        min_len = min(len(cam1_images), len(cam2_images))
        cam1_images = cam1_images[:min_len]
        cam2_images = cam2_images[:min_len]

    paired_images = list(zip(cam1_images, cam2_images))

    # 4. 标定摄像头1
    print("\n步骤3: 标定摄像头1")
    calibrator.calibrate_single_camera(cam1_images, "camera1")

    # 5. 标定摄像头2
    print("\n步骤4: 标定摄像头2")
    calibrator.calibrate_single_camera(cam2_images, "camera2")

    # 6. 立体标定
    print("\n步骤5: 立体标定")
    calibrator.calibrate_stereo(paired_images)

    # 7. 保存标定结果
    print("\n步骤6: 保存标定结果")
    calibrator.save_calibration("calibration.json")

    print("\n标定流程完成!")
    return calibrator


if __name__ == "__main__":
    run_calibration_pipeline()