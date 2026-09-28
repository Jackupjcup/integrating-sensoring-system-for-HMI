import numpy as np
import matplotlib.pyplot as plt
from IMUProcessing.MPU6050_utils import load_config
import torch

class SpatialMask:
    def __init__(self, device,
                 config_path=r"D:/blender_workplace/modules/hand_pose_mediapipe/mediapipe_config.json",
                 ROWS=14, COLS=10):

        self.media_config = load_config(config_path)
        self.rows_num = ROWS
        self.cols_num = COLS

        self.spatial_mask_param_dict = self.media_config["spatial_mask_params"]
        self.mask_keys_list = list(self.spatial_mask_param_dict.keys())
        self.masks_num = None
        self.spatial_mask = self.build_spatial_mask()  # type is {}

        self.device = device
        self.masks_tensor = self.masks_to_tensor()  # type is {}


    def build_spatial_mask(self):
        self.masks_num = len(self.spatial_mask_param_dict.keys())
        spatial_mask = {}

        for i, param in enumerate(self.spatial_mask_param_dict.keys()):
            mask = np.ones((self.rows_num, self.cols_num)) * 0.6

            red_region_coords = np.array(self.spatial_mask_param_dict[param]["red_region"])
            yellow_region_coords = np.array(self.spatial_mask_param_dict[param]["yellow_region"])

            if red_region_coords.shape != (0, ):

                rows, cols = zip(*red_region_coords)
                mask[list(rows), list(cols)] = 1

                rows, cols = zip(*yellow_region_coords)
                mask[list(rows), list(cols)] = 0.85

            spatial_mask[param] = mask

        return spatial_mask

    def masks_to_tensor(self):
        tensor_dict = {}
        for key in self.mask_keys_list:
            mask_np = self.spatial_mask[key]
            # 转换为张量 + 放到指定设备（GPU/CPU） + 转为float32（匹配输入类型）
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=self.device)
            # 扩展维度：适配输入x的形状 [batch, 5, 14, 10]
            # mask_tensor shape: [1, 1, 14, 10]（batch维度+序列维度）
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            tensor_dict[key] = mask_tensor
        return tensor_dict


    def plot_single_spatial_mask(self, mask, min_value=0, max_value=1):
        j_coords, i_coords = np.meshgrid(np.arange(self.cols_num), np.arange(self.rows_num))

        scatter = plt.scatter(
            j_coords,
            i_coords,
            c=mask,
            # c = pressure_array_trans.copy().reshape(140, -1),  # 每个点的压力值（颜色）
            cmap='coolwarm',
            vmin=min_value,
            vmax=max_value,
            marker='s',
            s=800,  # 点的大小（可根据坐标调整）
            edgecolors='white',  # 点的边框（区分相邻点）
            linewidth=3,
        )

        # 添加侧边颜色条
        plt.colorbar(scatter, label='Pressure Intensity')

        plt.show()




if __name__ == "__main__":
    spatial_mask_params = SpatialMask(device='cpu')

    print(spatial_mask_params.media_config["spatial_mask_params"].keys())

    # spatial_mask_params.build_spatial_mask()

    spatial_mask_params.plot_single_spatial_mask(mask=spatial_mask_params.spatial_mask["MM"])

    # print(spatial_mask_params.mask_keys_list)

