import pandas as pd
import numpy as np
from IMUProcessing.MPU6050_utils import load_config
from scipy.ndimage import median_filter, gaussian_filter
from collections import deque
import json
import os
from datetime import datetime


class RealTimePressureDenoiser:
    """
    实时压力传感器降噪器
    特点：低延迟、轻量，仅缓存少量历史帧，适配实时采集场景
    """

    def __init__(self,
                 sensor_shape=(14, 10),
                 time_window=5,  # 时间平滑窗口（缓存最近n帧，建议3-7）
                 sequence_median_size=(3, 1, 1),  # 时间中值滤波核大小
                 gaussian_sigma=1):  # 空间高斯滤波sigma
        self.sensor_shape = sensor_shape
        self.time_window = time_window
        self.sequence_median_size = sequence_median_size
        self.gaussian_sigma = gaussian_sigma

        # 初始化历史帧缓存队列（固定长度，满了自动弹出最旧的帧）
        self.frame_buffer = deque(maxlen=self.time_window)
        self.sequence_median_length = self.sequence_median_size[0]
        self.median_buffer = deque(maxlen=self.sequence_median_length)

    def denoise_single_frame(self, raw_frame, denoise_threshold=-100000):
        """
        对单帧原始数据进行实时降噪（核心方法）
        参数: raw_frame - 单帧原始数据，shape=(14,10)
        返回: denoised_frame - 降噪后的单帧数据，shape=(14,10)
        """
        # 步骤1：校验输入形状
        if raw_frame.shape != self.sensor_shape:
            raise ValueError(f"输入帧形状错误，应为{self.sensor_shape}，实际为{raw_frame.shape}")

        # 步骤2：转换数据类型（避免整数溢出/精度问题）
        frame = raw_frame.astype(np.float32)

        # 步骤3：Consider the threshold of denoising
        denoising_idx = np.where(frame > denoise_threshold)

        # 步骤4：空间去噪（无延迟，优先处理）
        # 中值滤波：去除脉冲噪声（传感器突然的大值/小值）
        # frame[denoising_idx] = median_filter(frame[denoising_idx], size=self.sequence_median_size)
        self.median_buffer.append(frame)
        if len(self.frame_buffer) == self.sequence_median_length:
            stack = np.array(self.median_buffer)
            frame = median_filter(stack, size=self.sequence_median_size)[1]
        # 高斯滤波：平滑空间分布，保留相邻传感器的关联性
        frame[denoising_idx] = gaussian_filter(frame[denoising_idx], sigma=self.gaussian_sigma)

        # 步骤5：时间平滑（低延迟，基于缓存的历史帧）
        # 将当前空间去噪后的帧加入缓存
        self.frame_buffer.append(frame)
        # 对缓存中的所有帧做均值平滑（窗口未满时，用已有帧的均值）
        # denoised_frame = np.mean(list(self.frame_buffer), axis=0)

        # 步骤6：限幅（可选，去除异常值，比如超出合理范围的噪声）
        denoised_frame = np.clip(frame, a_min=-1000, a_max=None)  # 压力值非负

        return denoised_frame.flatten()

    def reset_buffer(self):
        """重置历史帧缓存（比如采集场景切换时）"""
        self.frame_buffer.clear()
        print("历史帧缓存已重置！")


class PalmPointsClass:
    def __init__(self, rows_number, cols_number, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)

        # real-time denoiser
        self.denoiser = RealTimePressureDenoiser(sensor_shape=(14, 10))

        self.rows_num = rows_number
        self.cols_num = cols_number
        self.points_num = self.rows_num * self.cols_num
        self.denoise_sampling_num = 750
        self.whether_denoise = True

        # the accumulated pressure data
        self.cols_name = np.arange(0, self.points_num, 1)
        self.max_collected_frames = 500  # 限制最大存储帧数，防止内存无限增长
        # 使用 deque 替代 DataFrame 存储，避免 pandas 追加性能问题
        self.collected_points_deque = deque(maxlen=self.max_collected_frames)
        self.collected_points_df = pd.DataFrame(columns=self.cols_name)  # 仅在需要导出时使用
        self.pressure_data_sequence = deque(maxlen=self.denoise_sampling_num)
        # self.press_sensor_zero_bias_np = np.zeros(self.points_num)
        self.press_sensor_zero_bias_np = self.config["PlaneHandPressure_Display"]["ZeroBias"]

        # Dynamic zero bias
        self.dynamic_num_frame = 5

        # latest pressure data
        self.latest_pressure_data = np.zeros(self.points_num) # SHAPE OF (1, ROWS * COLS) LIKES (1, 140)
        self.latest_points_pressure_size = (1, self.points_num) # SHAPE OF (1, ROWS * COLS) LIKES (1, 140)
        self.raw_pressure_data = np.zeros(self.points_num)

        self.latest_pressure_data_table = self.latest_pressure_data.copy().reshape(self.rows_num, self.cols_num)

        self.collect_counts = 0
        self.buffer_length = 10 # useless
        self.MAX_VALUE = 1000


        # Blender Display : IT NEEDS POINTS SHAPE OF (1, ROWS * COLS) LIKES (1, 140)
        self.blender_points_id = np.array(self.config["Blender"]["HandMesh_ID"]).flatten() # shape of (1, 14)
        self.blender_points_id_shape = self.blender_points_id.shape
        self.finger_bones_dic = self.init_finger_bones()
        self.Metacarpals = self.config["Blender"]["Metacarpals"]
        #### Collect *positions* of compressed points
        self.all_points_position_df = pd.DataFrame(columns=["dt", "point_idx", "x", "y", "z"])
        #### Locate the *bending knuckle*
        self.bending_cols = np.zeros(self.cols_num, dtype=bool)
        self.bending_rows = np.zeros(self.rows_num, dtype=bool)
        #### Collect *angles* of fingers bending
        self.bending_angles_df, self.latest_bending_angles_dic = self.init_finger_bending_angles_df_dic()
        self.bending_knuckles_indicate_names = self.bending_angles_df.columns.values
        self.bending_knuckles_indicate = np.zeros(len(self.bending_knuckles_indicate_names), dtype=bool)
        self.bending_knuckles_indicate_df = pd.DataFrame(columns=self.bending_knuckles_indicate_names)
        self.bending_knuckles_IDs = None
        self.bending_knuckles_count = 0

        # Plane Display Pressure : IT NEEDS POINT SHAPE OF (1, ROWS * COLS) LIKES (1, 140)
        self.plane_pressure_points_coords = self.config["PlaneHandPressure_Display"]["PlaneCoords"] # shape of (1, 140)

        # Test Flag
        self.tst_flag = True
        self.calibration_flag = False

    def init_finger_bones(self):
        fingers = {"index": self.config["Blender"]["Finger_Bones"]["index"],
                   "middle": self.config["Blender"]["Finger_Bones"]["middle"],
                   "ring": self.config["Blender"]["Finger_Bones"]["ring"],
                   "pinky": self.config["Blender"]["Finger_Bones"]["pinky"],
                   "thumb": self.config["Blender"]["Finger_Bones"]["thumb"]}
        return fingers
    def init_finger_bending_angles_df_dic(self):
        bending_angles_df = pd.DataFrame()
        latest_bending_angles_dic = {}
        for f_chain in self.finger_bones_dic.values():
            for bone_name in f_chain:
                if bone_name != "TMetacarpal":
                    bending_angles_df[bone_name] = np.nan
                    latest_bending_angles_dic[bone_name] = 0

        return bending_angles_df, latest_bending_angles_dic

    def transform_latest_pressure_data(self, latest_pressure_data):
        if latest_pressure_data.shape == (self.rows_num, self.cols_num):
            self.latest_pressure_data = self.process_latest_pressure_data(np.array(latest_pressure_data).flatten())
            self.latest_pressure_data_table = self.latest_pressure_data.copy().reshape(self.rows_num, self.cols_num)
            self.pressure_data_sequence.append(self.latest_pressure_data_table)
            if not self.tst_flag:
                self.locate_knuckle()
                # self.tst_flag = True
            # 使用 deque 存储，O(1) 复杂度，避免 DataFrame 追加性能问题
            self.collected_points_deque.append(self.latest_pressure_data.copy())
            self.collect_counts += 1
        else:
            print(f"require shape of latest pressure data is {[self.rows_num, self.cols_num]}, but received shape of {latest_pressure_data.shape}")

    def process_latest_pressure_data(self, latest_pressure_data): # latest_pressure_data shape is [1, Total]
        if self.collect_counts % (8*self.dynamic_num_frame) == 0 and not self.calibration_flag:
            self.dynamic_adjusting_bias(latest_pressure_data)
        latest_pressure_data = latest_pressure_data #- self.press_sensor_zero_bias_np
        self.raw_pressure_data = latest_pressure_data

        if self.whether_denoise and not self.calibration_flag:
            latest_pressure_data = self.denoiser.denoise_single_frame(latest_pressure_data.reshape(14, 10))
        return latest_pressure_data

    def dynamic_adjusting_bias(self, pressure_data_np, mean_threshold=15, var_threshold=5000):
        mean_vals = np.mean(pressure_data_np - self.press_sensor_zero_bias_np)
        var_vals = np.var(pressure_data_np - self.press_sensor_zero_bias_np)

        if (np.abs(mean_vals) > mean_threshold) and (var_vals <= var_threshold):
            try:

                self.press_sensor_zero_bias_np = pressure_data_np
                print(f"mean vals: {int(mean_vals)}, var vals: {int(var_vals)}. " + "Modify Zero Bias")

            except Exception as e:
                print(f"未知错误：{e}")
        else:
            print(f"mean vals: {int(mean_vals)}, var vals: {int(var_vals)}. " + "Maintain Zero Bias")


    def high_freq_filter(self, latest_pressure_data, single_threshold=50): # I may need the variance
        if len(self.collected_points_deque) > self.buffer_length:
            self.latest_buffer = np.array(list(self.collected_points_deque)[-self.buffer_length:])
            diff_buffer_sum = abs(self.latest_buffer - latest_pressure_data).sum(axis=0) # Shape [1, total]
            latest_pressure_data[np.where(diff_buffer_sum > single_threshold * self.buffer_length)] = 0
        return latest_pressure_data

    def locate_knuckle(self, change_rate_threshold=100, col_1st_threshold=100, col_2nd_threshold=200):
        # pressure points table -> rows and cols connection
        # In Rows: When specific column has dramatic change, it means the specific finger
        # In Cols: Doubly check the range of all columns in specific row i and row (13 - i)
        #          Row [1, 12] -> M
        #          Row [2, 11] -> P
        #          Row [0, 13] -> I
        '''in rows'''
        latest_pressure_data_table_ = self.latest_pressure_data_table.copy()
        change_rates_np = np.diff(latest_pressure_data_table_, axis=1)
        pos_max_change_rate = np.max(change_rates_np, axis=1)
        neg_max_change_rate = np.min(change_rates_np, axis=1)
        # target_rows = np.where((pos_max_change_rate > change_rate_threshold) & (neg_max_change_rate < -change_rate_threshold))[0]
        combined_condition = np.logical_and(
            pos_max_change_rate > change_rate_threshold,
            neg_max_change_rate < -change_rate_threshold
        )
        target_rows = np.where(combined_condition)[0]

        lift_idx = np.where(change_rates_np >= change_rate_threshold)
        drop_idx = np.where(change_rates_np >= -change_rate_threshold)

        lift_count = 0
        drop_count = 0

        # print(change_rates_np)
        # print(change_rates_np.shape)
        # print(pos_max_change_rate)
        # print(neg_max_change_rate)
        # print(np.where(combined_condition))
        # print(target_rows)

        for i in target_rows:
            pos_neg_pairs = []
            pair_0 = None
            pair_1 = None
            lift_num = (lift_idx[0] == i).sum()
            drop_num = (drop_idx[0] == i).sum()


            cols_cat = np.hstack((lift_idx[1][lift_count: lift_count + lift_num],
                                  drop_idx[1][drop_count: drop_count + drop_num]))
            left_edge_flag = False

            for j in np.sort(cols_cat):
                index = np.where(cols_cat == j)[0][0]
                if index < lift_num:
                    if pair_0 is None and j != np.sort((cols_cat))[-1]:
                        pair_0 = cols_cat[index]
                    elif pair_0 is None and j == np.sort((cols_cat))[-1]:
                        pair_0 = cols_cat[index]
                        pairs = self.diff_idx_to_data_idx([pair_0, pair_1])
                        pos_neg_pairs.append(pairs)
                        pair_0 = None
                    elif pair_0 is not None and left_edge_flag:
                        pair_1 = pair_0
                        pairs = self.diff_idx_to_data_idx([pair_0, pair_1])
                        pos_neg_pairs.append(pairs)
                        pair_0 = cols_cat[index]
                        pair_1 = None
                if index >= lift_num:
                    if pair_0 is not None:
                        pair_1 = cols_cat[index]
                        pairs = self.diff_idx_to_data_idx([pair_0, pair_1])
                        pos_neg_pairs.append(pairs)
                        pair_0 = None
                        pair_1 = None
                    elif pair_0 is None and cols_cat[index] == 0:
                        pair_0 = cols_cat[index]
                        left_edge_flag = True

            lift_count+=lift_num
            drop_count+=drop_num

        # print(self.bending_cols)
        # print(target_rows)

        '''in cols'''
        cols_data_range = np.max(latest_pressure_data_table_, axis=1) - np.min(latest_pressure_data_table_, axis=1)

        cols_range_satisfied = np.where(cols_data_range > col_1st_threshold)[0]
        if np.isin(1, cols_range_satisfied) and np.isin(12, cols_range_satisfied):
            self.bending_rows[1] = True
            self.bending_rows[12] = True
        if np.isin(2, cols_range_satisfied) and np.isin(11, cols_range_satisfied):
            self.bending_rows[2] = True
            self.bending_rows[11] = True

        # print(cols_range_satisfied)
        # print(self.bending_rows)
        # print('\n')

        '''Summary'''
        rows, cols = np.meshgrid(np.where(self.bending_rows)[0], np.where(self.bending_cols)[0], indexing='ij')
        self.bending_knuckles_IDs = np.stack([rows.ravel(), cols.ravel()], axis=1)
        # print(self.bending_knuckles_IDs)
        self.knuckles_indices_to_bones()
        print(self.bending_knuckles_indicate_names)
        print(self.bending_knuckles_indicate)
        self.reset_latest_bending_knuckles_indicate()
        # print(self.bending_knuckles_indicate_df.iloc[-1, :].values)

    def knuckles_indices_to_bones(self):
        if [1, 6] in self.bending_knuckles_IDs or [1, 7] in self.bending_knuckles_IDs or [12, 6] in self.bending_knuckles_IDs or [12, 7] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "IM")[0]] = True
        if [2, 6] in self.bending_knuckles_IDs or [2, 7] in self.bending_knuckles_IDs or [11, 6] in self.bending_knuckles_IDs or [11, 7] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "IP")[0]] = True
        if [1, 4] in self.bending_knuckles_IDs or [1, 5] in self.bending_knuckles_IDs or [12, 4] in self.bending_knuckles_IDs or [12, 5] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "MM")[0]] = True
        if [2, 4] in self.bending_knuckles_IDs or [2, 5] in self.bending_knuckles_IDs or [11, 4] in self.bending_knuckles_IDs or [11, 5] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "MP")[0]] = True
        if [1, 2] in self.bending_knuckles_IDs or [1, 3] in self.bending_knuckles_IDs or [12, 2] in self.bending_knuckles_IDs or [12, 3] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "RM")[0]] = True
        if [2, 2] in self.bending_knuckles_IDs or [2, 3] in self.bending_knuckles_IDs or [11, 2] in self.bending_knuckles_IDs or [11, 3] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "RP")[0]] = True
        if [1, 0] in self.bending_knuckles_IDs or [1, 1] in self.bending_knuckles_IDs or [12, 0] in self.bending_knuckles_IDs or [12, 1] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "PM")[0]] = True
        if [2, 0] in self.bending_knuckles_IDs or [2, 1] in self.bending_knuckles_IDs or [11, 0] in self.bending_knuckles_IDs or [11, 1] in self.bending_knuckles_IDs:
            self.bending_knuckles_indicate[np.where(self.bending_knuckles_indicate_names == "PP")[0]] = True
        # if [1, 6] in self.bending_knuckles_IDs or [1, 7] in self.bending_knuckles_IDs or [12, 6] in self.bending_knuckles_IDs or [12, 7] in self.bending_knuckles_IDs:
        #     self.bending_knuckles_indicate['MM'] = True
        self.bending_knuckles_indicate_df.loc[self.bending_knuckles_count, :] = self.bending_knuckles_indicate
        self.bending_knuckles_count += 1

    def reset_latest_bending_knuckles_indicate(self):
         self.bending_knuckles_indicate = np.zeros(len(self.bending_knuckles_indicate_names), dtype=bool)
         self.bending_cols = np.zeros(self.cols_num, dtype=bool)
         self.bending_rows = np.zeros(self.rows_num, dtype=bool)

    def diff_idx_to_data_idx(self, pairs):
        if pairs[0] < pairs[1]:
            pairs[0] += 1
        self.bending_cols[pairs[0]] = True
        self.bending_cols[pairs[1]] = True
        return pairs


    def save_collected_points_df(self, filepath='collected_points_pressure.csv'):
        """将 deque 中的数据导出为 CSV 文件"""
        if len(self.collected_points_deque) > 0:
            self.collected_points_df = pd.DataFrame(list(self.collected_points_deque), columns=self.cols_name)
            self.collected_points_df.to_csv(filepath, index=False)
            print(f"Saved {len(self.collected_points_deque)} frames to {filepath}")
        else:
            print("No data to save")



        
        
        


    
    
    


if __name__ == '__main__':

    PnF = PalmPointsClass(14, 10, config_path="D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")

    print(PnF.bending_angles_df)
    

