import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from IMUProcessing.MPU6050_utils import load_config


palm_points_config = load_config("D:/blender_workplace/modules/PalmPoints_and_FingerBend/PalmPointsConfig.json")
sensor_coords = np.array(palm_points_config["PlaneHandPressure_Display"]["PlaneCoords"])

def raw_data_to_true_img(pressure,
                         list=[0, 13, 1, 12, 2, 11, 3, 10, 4, 9, 5, 8, 6, 7],
                         max_row=13):
    pressure_real = np.zeros(pressure.shape)

    for i, row in enumerate(list):
        pressure_real[max_row - i] = pressure[row]

    return pressure_real

def plot_pressure_heat(pressure_data, plot_name, whether_square_shape=False):
    if pressure_data.shape[0] == 10:
        fig, axes = plt.subplots(2, 5, figsize=(30, 10))
        fig.suptitle(f"{plot_name}", fontsize=16, y=0.98)

        axes_flat = axes.flatten()

        for idx in range(pressure_data.shape[0]):
            ax = axes_flat[idx]
            pressure_data_np = pressure_data.iloc[idx].values
            im = plot_single_pressure(pressure_data_np, whether_square_shape=whether_square_shape, ax=ax)

        cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])  # [left, bottom, width, height]
        fig.colorbar(
            im,
            cax=cbar_ax,
            label='Pressure Intensity',
            shrink=0.8
        )

        # 调整子图间距，避免重叠
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出颜色条的位置
        plt.show()

def plot_single_pressure(pressure_data, whether_square_shape, ax):
    if whether_square_shape:
        im = ax.imshow(pressure_data.reshape(ROWS, COLS), cmap="coolwarm", aspect="auto", vmin=0, vmax=MAX_VALUE)
        # plt.colorbar(im, label='Pressure Intensity')

        texts = []
        for i in range(ROWS):
            for j in range(COLS):
                text = ax.text(j, i, "", ha="center", va="center",
                               fontsize=10, fontweight='bold')
                texts.append(text)

        # ax.set_title("Real-time Pressure Matrix Values")
        ax.set_xticks(np.arange(COLS))
        ax.set_yticks(np.arange(ROWS))

        for idx in range(TOTAL_NUM):
            val = pressure_data[idx]
            texts[idx].set_text(f"{int(pressure_data[idx])}")

            # 智能调整文字颜色：如果背景太红/太深，文字改为白色
            if val > MAX_VALUE * 0.6:
                texts[idx].set_color("white")
            else:
                texts[idx].set_color("black")

    else:
        im = ax.scatter(
            [coord[0] for coord in sensor_coords],  # 所有点的x坐标
            [-coord[1] for coord in sensor_coords],
            c=pressure_data,  # 每个点的压力值（颜色）
            cmap='coolwarm',
            vmin=0,
            vmax=MAX_VALUE,
            marker='s',
            s=200,  # 点的大小（可根据坐标调整）
            edgecolors='white',  # 点的边框（区分相邻点）
            linewidth=0.5,
        )

        # # 添加侧边颜色条
        # plt.colorbar(im, label='Pressure Intensity')

        texts = []
        for sensor_coord in sensor_coords:
            x, y = sensor_coord
            text = ax.text(x, -y, "", ha="center", va="center",
                           fontsize=6, fontweight='bold')
            texts.append(text)

        # ax.set_title("Real-time Pressure Matrix Values")
        ax.set_xticks(np.arange(COLS))
        ax.set_yticks(np.arange(ROWS))

        for idx in range(TOTAL_NUM):
            val = pressure_data[idx]
            texts[idx].set_text(f"{int(pressure_data[idx])}")

            # 智能调整文字颜色：如果背景太红/太深，文字改为白色
            if val > MAX_VALUE * 0.6:
                texts[idx].set_color("white")
            else:
                texts[idx].set_color("black")
    return im



def plot_row_and_col_connection(pressure_data_np, rows=14, cols=10):
    pressure_data_np = pressure_data_np.reshape(rows, cols)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    for i in range(rows):
        axs[0].plot(pressure_data_np[i, :], label=f"row {i}")
    axs[0].legend()

    for j in range(cols):
        axs[1].plot(pressure_data_np[:, j], label=f"col {j}")
    axs[1].legend()
    plt.show()

class Analysis_Freq:
    def __init__(self, pressure_data_np, rows=14, cols=10):
        self.pressure_data_np = pressure_data_np # Shape of pressure_data_np is [samples num, points num]
        self.rows = rows
        self.cols = cols
        self.Fs = 100
        self.N = self.pressure_data_np.shape[0]
        self.t = np.linspace(0, (self.N - 1) / self.Fs, self.N)

    def FFT_(self, point_idx, whether_plot_freq_domain=True, whether_plot_freq_heat_map=True):
        dominant_freq_list = []
        max_amplitude_list = []
        for i in range(self.rows * self.cols):
            data = self.pressure_data_np[:, i]
            fft_result = np.fft.fft(data)
            amplitude = np.abs(fft_result) / self.N
            amplitude = 2 * amplitude[:self.N // 2]
            freq = np.fft.fftfreq(self.N, 1 / self.Fs)[:self.N // 2]

            non_zero_idx = freq > 0  # 过滤0Hz
            max_amp_idx = np.argmax(amplitude[non_zero_idx])
            dominant_freq = freq[non_zero_idx][max_amp_idx]
            dominant_freq_list.append(dominant_freq)
            max_amplitude = amplitude[non_zero_idx][max_amp_idx]
            max_amplitude_list.append(max_amplitude)

            if i == point_idx:
                data_flag, freq_flag, amplitude_flag = data, freq, amplitude

        print(f"采样频率 Fs = {self.Fs} Hz")
        print(f"频率分辨率 = {self.Fs / self.N:.4f} Hz")
        print(f"主导频率 = {dominant_freq_list[point_idx]:.2f} Hz (振幅 = {max_amplitude_list[point_idx]:.2f})")

        if whether_plot_freq_heat_map:
            fig1, ax1 = plt.subplots(figsize=(8, 10))
            dominant_freq_reshaped = np.array(dominant_freq_list).reshape((self.rows, self.cols))
            dominant_freq_reshaped = raw_data_to_true_img(dominant_freq_reshaped)
            im1 = ax1.imshow(dominant_freq_reshaped, cmap='coolwarm')
            cbar = fig1.colorbar(im1, ax=ax1)
            cbar.set_label("Dominant Frequency")

            for i in range(dominant_freq_reshaped.shape[0]):
                for j in range(dominant_freq_reshaped.shape[1]):
                    # 在每个格子中心添加数值，保留2位小数
                    text = ax1.text(j, i, f'{dominant_freq_reshaped[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=10)

            plt.show()

        if whether_plot_freq_domain:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 时域图（原始折线图）
            ax1.plot(self.t, data_flag, 'b-', label='Raw Data')
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('amplitude')
            ax1.set_title(f'Time Domain ({self.N} points, Fs {self.Fs}Hz)')
            ax1.legend()
            ax1.grid(True)

            # 频域图（FFT振幅谱）
            ax2.plot(freq_flag, amplitude_flag, 'r-', label='FFT')
            ax2.axvline(x=dominant_freq_list[point_idx], color='g', linestyle='--',
                        label=f'Dominant Freq: {dominant_freq_list[point_idx]:.2f}Hz')
            ax2.set_xlabel('Freq (Hz)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Freq Domain (FFT Analysis)')
            ax2.set_xlim(0, self.Fs / 2)  # 只显示有效频率范围（奈奎斯特频率=Fs/2）
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()


    def plot_pressure_time_sequence(self, row_idx, col_idx, whether_display): # The Freq Connection in Rows and Cols
        if whether_display:
            plt.subplot(1, 2, 1)
            for i in range(self.cols):
                data = self.pressure_data_np[:, row_idx * self.cols + i]
                plt.plot(data, label=f"col {i}")
            plt.legend()
            plt.title(f"Row {row_idx} Pressure Map")

            plt.subplot(1, 2, 2)
            for i in range(self.rows):
                data = self.pressure_data_np[:, (self.cols - 1) * i + col_idx]
                plt.plot(data, label=f"row {i}")
            plt.legend()
            plt.title(f"Col {col_idx} Pressure Map")
            plt.show()


def load_pressure_data_mean(folder_path, total_points=140):

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    csv_files.sort()
    samples_df = pd.DataFrame(np.zeros((len(csv_files), total_points)))

    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(folder_path, file), header=None)
        samples_df.loc[i, :] = np.mean(df.values, axis=0)
        i += 1

    return samples_df

def load_pressure_data(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    csv_files.sort()

    df_list = []
    # label_idx = folder_path[-1]

    number_str = folder_path.split("_")[-1]
    label_idx = int(number_str)

    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(folder_path, file), header=None)
        df["label"] = int(label_idx) - 1
        df_list.append(df)

    return df_list

def raw_pressure_to_cnn_img(hand_gesture_list, fp, whether_save_as_npz=True):

    if os.path.exists(fp + "cnn_data.npz") and os.path.exists(fp + "merged_data.csv"):
        merged_df = pd.read_csv(fp + "merged_data.csv")
        loaded_data = np.load(fp + "cnn_data.npz")
        images_np = loaded_data["images"]
        labels_np = loaded_data["labels"]
        print('cnn_data.npz and merged_data.csv have been existed')
    else:
        df_list = []
        for idx, hand_gesture in enumerate(hand_gesture_list):
            csv_path = fp + hand_gesture

            df_list.extend(load_pressure_data(csv_path))

        merged_df = pd.concat(df_list, ignore_index=True)

        features_np = merged_df.iloc[:, :-1].values
        labels_np = merged_df.iloc[:, -1].values

        images_np = features_np.reshape(-1, 14, 10)
        images_np = np.expand_dims(images_np, axis=1)

        images_np = images_np.astype(np.float32)
        labels_np = labels_np.astype(np.int64)

        if whether_save_as_npz:
            np.savez_compressed(
                fp + "cnn_data.npz",
                images=images_np,
                labels=labels_np,
                allow_pickle=False
            )
            print("图像+标签已保存为cnn_data.npz")

            merged_df.to_csv(fp + "merged_data.csv", index=False)
            print("merged data is saved as merged_data.csv")

    return merged_df, images_np, labels_np

def merge_hand_gesture_data(files_path_list):
    '''
    file_path_list:
        fp_1 = f"D:/blender_workplace/modules/Functions/Hand_Gesture/20260115-14/cnn_data.npz"
        fp_2 = f"D:/blender_workplace/modules/Functions/Hand_Gesture/20260115-21/cnn_data.npz"
        fp_list = [fp_1, fp_2]
    '''
    num_files = len(files_path_list)
    merged_imgs = []
    merged_lbs = []
    for i, file in enumerate(files_path_list):
        loaded_data = np.load(file)
        img_np = loaded_data["images"]
        labels_np = loaded_data["labels"]

        if i == 0:
            merged_imgs = img_np
            merged_lbs = labels_np
        else:
            merged_imgs = np.concatenate([merged_imgs, img_np], axis=0)
            merged_lbs = np.concatenate([merged_lbs, labels_np], axis=0)

    return merged_imgs, merged_lbs



if __name__ == "__main__":
    COUNT = 0
    ROWS, COLS = 14, 10
    TOTAL_NUM = ROWS * COLS
    MAX_VALUE = 1000
    
    # HAND_GESTURE = "HandGesture_5"
    # HAND_GESTURE = "Empty_Long_Without_Dynamic_Adj"

    '''Raw pressure(gesture) data to images(.npz) for CNN training'''
    HAND_GESTURE_LIST = [f"object_{i}" for i in range(1, 8)]
    folder_path = "D:/blender_workplace/modules/Functions/objects_recognition/Objects_Collection/20260329-11/"

    merged_data, images_np, labels_np = raw_pressure_to_cnn_img(HAND_GESTURE_LIST, folder_path, whether_save_as_npz=True)
    print(images_np.shape)
    print(labels_np.shape)
    print(f"标签唯一值: {np.unique(labels_np)}")


    '''Merge .npz together to a whole .npz file'''
    fp_list = [
        "D:/blender_workplace/modules/Functions/objects_recognition/Objects_Collection/20260329-11/cnn_data.npz"
               ]
    # fp_list = [f"D:/blender_workplace/modules/Functions/Hand_Gesture/20260310-12/cnn_data.npz",
    #            f"D:/blender_workplace/modules/Functions/Hand_Gesture/20260310-18/cnn_data.npz"]

    merged_images, merged_labels = merge_hand_gesture_data(fp_list)
    print(f"Total {len(merged_labels)} Samples")
    np.savez_compressed(
        "merged_data.npz",
        images=merged_images,
        labels=merged_labels,
        allow_pickle=False
    )



    '''Analysis of Empty Gesture's Frequency'''
    # log_date_path = "D:/blender_workplace/modules/Functions/Hand_Gesture/20260326-21/"
    # HAND_GESTURE = "donot_ideal_gesture_4"
    # csv_folder_path = log_date_path + HAND_GESTURE
    # df = pd.read_csv(csv_folder_path + "/20260326_210529.csv")
    # ana_freq = Analysis_Freq(df.values, ROWS, COLS)
    # ana_freq.plot_pressure_time_sequence(0, 0, True)
    # ana_freq.FFT_(point_idx=0, whether_plot_freq_domain=True, whether_plot_freq_heat_map=True)
    #
    # ''' Illustrate the relationship in rows and columns '''
    # # plot_row_and_col_connection(samples_df.iloc[1].values)


    ''' Illustrate the heat map of palm pressure data '''
    # # HAND_GESTURE_LIST = [f"HandGesture_{i}" for i in range(1, 6)]
    # HAND_GESTURE_LIST = [f"ideal_gesture_{i}" for i in range(1, 12)]
    # HAND_GESTURE_dir = "ideal_gesture_"
    # HAND_GESTURE = HAND_GESTURE_dir + f"{8}"
    # log_date_path = "D:/blender_workplace/modules/Functions/Hand_Gesture/20260310-18/"
    # csv_folder_path =  log_date_path + HAND_GESTURE
    # bedrock_path = log_date_path + HAND_GESTURE_dir + f"1"
    #
    # samples_df = load_pressure_data_mean(csv_folder_path)  # shape of [num_of_recorder, num_od_points(140)]
    # bedrock_df = load_pressure_data_mean(bedrock_path)
    # # label_mean = np.mean(samples_df.values, axis=0)
    # # samples_mean = np.mean(samples_df.values, axis=1)
    #
    # print(samples_df.shape)
    # print(bedrock_df.shape)
    # # print(samples_mean.shape)
    # # print(label_mean.shape)
    #
    # plot_pressure_heat(samples_df.iloc[0: 10], HAND_GESTURE, whether_square_shape=False)
    #
    # # fig, ax = plt.subplots(figsize=(10, 10))
    # # im = plot_single_pressure(samples_df.iloc[0].values, whether_square_shape=False, ax=ax)
    # # plt.show()
    #
    # # plot_pressure_heat(samples_df.iloc[0].values, HAND_GESTURE, whether_square_shape=False)
    # # plot_pressure_heat(samples_mean, HAND_GESTURE, whether_square_shape=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    