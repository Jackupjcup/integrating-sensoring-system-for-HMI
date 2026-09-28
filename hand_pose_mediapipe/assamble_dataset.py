import numpy as np
import pandas as pd

def load_df_as_np(data_path, attr="csv"):
    if data_path[-3:] == attr:
        df = pd.read_csv(data_path)
        return np.array(df.iloc[:, :].values)
    else:
        print("attraction of data is not .csv")
        return None

def add_noise_on(data, low=20, high=50):

    noisy_data = data[:, 1:].copy()

    for i in range(data.shape[0]):
        frame = noisy_data[i]  # 当前帧 (14,10)

        # 每帧随机选 10 ~ 20 个位置
        num_noise = np.random.randint(10, 21)  # [10,20]

        indices = np.random.choice(140, size=num_noise, replace=False)

        # 生成 20~50 噪声
        noise = np.random.uniform(low, high, num_noise)

        # 只给选中的位置加噪声
        frame[indices] += noise

    data[:, 1:] = noisy_data

    return data

def load_angles_df_as_np(data_path, attr="csv"):
    if data_path[-3:] == attr:
        df = pd.read_csv(data_path)
        raw_nums = df.shape[0]
        df = df[df["hand_index"] != 1]
        final_nums = df.shape[0]
        print(f"{data_path}: Filter {raw_nums - final_nums} Left Hand")

        return np.array(df.iloc[:, :].values)
    else:
        print("attraction of data is not .csv")
        return None
def raw_data_to_true_img(pressure, list=[0, 13, 1, 12, 2, 11, 3, 10, 4, 9, 5, 8, 6, 7], max_row=13):
    pressure_real = np.zeros(pressure.shape)

    for i, row in enumerate(list):
        pressure_real[max_row - i] = pressure[row]

    return pressure_real


def align_with(camera_data, sensor_data):
    whether_interpolation = True
    whether_nearest = False

    sensor_times = sensor_data[:,0]

    dataset = []
    pressure_df = []

    for cam in camera_data:

        t = cam[0]
        angles = cam[4:]

        idx = np.searchsorted(sensor_times, t)

        if whether_interpolation:

            # 处理边界情况
            if idx == 0 or idx >= len(sensor_times):
                continue

            t1 = sensor_times[idx-1]
            t2 = sensor_times[idx]

            p1 = sensor_data[idx-1,1:]
            p2 = sensor_data[idx,1:]

            w = (t - t1)/(t2 - t1)

            pressure = (1-w)*p1 + w*p2
            pressure = pressure.reshape(14,10)
            pressure = raw_data_to_true_img(pressure)

        elif whether_nearest:
            # 处理边界情况
            if idx == 0:
                closest_idx = 0
            elif idx >= len(sensor_times):
                closest_idx = len(sensor_times) - 1
            else:
                # 比较前后两个传感器点，选**时间更近**的那一个
                t_left = sensor_times[idx - 1]
                t_right = sensor_times[idx]
                if t - t_left < t_right - t:
                    closest_idx = idx - 1
                else:
                    closest_idx = idx

            pressure = sensor_data[closest_idx, 1:]

        dataset.append({
            "pressure": pressure,
            "angles": angles
            # "timestamp": t
        })

    return dataset


import numpy as np


def align_sensor_camera(camera_data, sensor_data):
    '''
    对齐相机数据与传感器数据，为每条传感器数据匹配时间最近的相机帧
    以传感器数据为主遍历（传感器帧数更少），通过时间戳找到最近的相机帧进行配对
    处理后返回包含压力图像和角度信息的数据集
    '''
    # 提取相机和传感器的时间戳
    cam_times = camera_data[:, 0]
    sensor_times = sensor_data[:, 0]

    dataset = []

    # 遍历每一条传感器数据（因为传感器更少，以此为主）
    for i, sensor in enumerate(sensor_data):
        t_sensor = sensor[0]  # 当前传感器时间
        pressure_raw = sensor[1:]  # 传感器压力数据

        # 找到离 sensor 最近的 camera 帧
        idx = np.searchsorted(cam_times, t_sensor)

        # 处理边界
        if idx == 0:
            closest_cam_idx = 0
        elif idx >= len(cam_times):
            closest_cam_idx = len(cam_times) - 1
        else:
            # 比较左右两个相机帧，选时间最近的
            t_left = cam_times[idx - 1]
            t_right = cam_times[idx]
            if t_sensor - t_left < t_right - t_sensor:
                closest_cam_idx = idx - 1
            else:
                closest_cam_idx = idx

        # 取出最近的相机帧数据
        cam = camera_data[closest_cam_idx]
        angles = cam[4:]

        # 压力数据格式不变
        pressure = pressure_raw.reshape(14, 10)
        pressure = raw_data_to_true_img(pressure)

        # 输出：1个sensor → 对应1个最近cam
        dataset.append({
            "pressure": pressure,
            "angles": angles
        })

    return dataset

def normalize_angle_data(angles_data_np, max_value=180, min_value=0, angle_idx=4):

    angles_data_np[:, angle_idx:] = np.clip(angles_data_np[:, angle_idx:], a_min=min_value, a_max=max_value)

    angles_data_np[:, angle_idx:] = angles_data_np[:, angle_idx:] / max_value

    return angles_data_np


def normalize_pressure_data(pressure_data_np, pressure_idx=1):

    pure_pressure_np = pressure_data_np[:, pressure_idx:].copy()
    # mean_pressure = np.mean(pure_pressure_np, axis=0)
    # std_pressure = np.std(pure_pressure_np, axis=0)

    mean_pressure = np.array(pure_pressure_np.mean())
    std_pressure = np.array(pure_pressure_np.std())

    pure_pressure_np_norm = (pure_pressure_np - mean_pressure) / (std_pressure + 1e-8)

    pressure_data_np[:, pressure_idx:] = pure_pressure_np_norm

    return pressure_data_np, mean_pressure, std_pressure

def get_pressure_data_distribution(pressure_data_np, max_value=500, min_value=-200, pressure_idx=1):
    pure_pressure_np = pressure_data_np[:, pressure_idx:].copy()
    pure_pressure_np = np.clip(pure_pressure_np, a_min=min_value, a_max=max_value)

    total = np.sum(pressure_data_np, axis=1, keepdims=True)

    total[total == 0] = 1.0

    pressure_distribution = pure_pressure_np / total

    pressure_data_np[:, pressure_idx:] = pressure_distribution

    return pressure_data_np

def save_mean_and_std(
        array1: np.ndarray,
        array2: np.ndarray,
        save_path: str,
        delimiter: str = ",",
        header: str = "mean,std"
) -> bool:
    """
    将两个长度为(140,)的NumPy数组存入TXT文件
    :param array1: 第一个数组（如压力数据，shape=(140,)）
    :param array2: 第二个数组（如角度数据，shape=(140,)）
    :param save_path: TXT文件保存路径（如 "data/frame_001.txt"）
    :param delimiter: 列分隔符（默认逗号，可选空格/制表符）
    :param header: 表头（默认"pressure_data,angle_data"）
    :return: 保存成功返回True，失败返回False
    """
    try:
        # ========== 1. 数据校验（避免格式错误） ==========
        # 检查是否为NumPy数组
        if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
            raise TypeError("输入必须是NumPy数组！")

        # 检查数组形状（必须是(140,)）
        # if array1.shape != (140,) or array2.shape != (140,):
        #     raise ValueError(f"数组形状必须为(140,)！当前：array1={array1.shape}, array2={array2.shape}")

        # 转换为浮点型（统一数据类型）
        array1 = array1.astype(np.float32)
        array2 = array2.astype(np.float32)

        # ========== 2. 拼接数据（按列拼接：140行×2列） ==========
        # 把两个(140,)的数组拼接成(140, 2)的二维数组
        combined_data = np.column_stack((array1, array2))

        # ========== 3. 写入TXT文件 ==========
        # fmt="%.4f"：保留4位小数，避免科学计数法
        np.savetxt(
            fname=save_path,
            X=combined_data,
            fmt="%.4f",
            delimiter=delimiter,
            header=header,
            comments=""  # 去掉表头前的#号
        )

        print(f"数据已成功保存到：{save_path}")
        print(f"数据形状：{combined_data.shape}")
        return True

    except Exception as e:
        print(f"保存失败：{e}")
        return False
import numpy as np


def read_two_140d_arrays_from_txt(
    file_path: str,
    delimiter: str = ",",
    skip_header: int = 1  # 跳过表头行（默认1行）
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    读取TXT文件中保存的两个(140,)的NumPy数组
    :param file_path: TXT文件路径（如 "data/frame_001.txt"）
    :param delimiter: 列分隔符（需与保存时一致，默认逗号）
    :param skip_header: 跳过的行数（保存时写了表头则设为1，无表头设为0）
    :return: (array1, array2) 两个(140,)的数组；读取失败返回(None, None)
    """
    try:
        # ========== 1. 读取TXT文件 ==========
        # 读取为(140, 2)的二维数组
        combined_data = np.loadtxt(
            fname=file_path,
            delimiter=delimiter,
            skiprows=skip_header,  # 跳过表头
            dtype=np.float32       # 与保存时的类型一致
        )

        # ========== 2. 数据校验 ==========
        # 检查读取后的数据形状是否为(140, 2)
        if combined_data.shape != (140, 2):
            raise ValueError(f"文件数据形状错误！预期(140,2)，实际{combined_data.shape}")

        # ========== 3. 拆分为两个(140,)的数组 ==========
        array1 = combined_data[:, 0]  # 第一列（压力数据）
        array2 = combined_data[:, 1]  # 第二列（角度数据）

        print(f"数据读取成功：{file_path}")
        print(f"array1形状：{array1.shape}，array2形状：{array2.shape}")
        return array1, array2

    except FileNotFoundError:
        print(f"读取失败：文件不存在 → {file_path}")
        return None, None
    except ValueError as e:
        print(f"读取失败：数据格式错误 → {e}")
        return None, None
    except Exception as e:
        print(f"读取失败：未知错误 → {e}")
        return None, None
    
def merge_all_data_as_dataset(dir, session_list, merge_list,
                              whether_save_every_dataset=True,
                              angles_data_name="angles_data.csv",
                              pressure_data_name="pressure_data.csv"):

    total_dataset = pd.DataFrame()
    mean_pressure_list = []
    std_pressure_list = []
    
    for i, session in enumerate(session_list):
        angles_data_path = dir + session + "/" + angles_data_name
        pressure_data_path = dir + session + "/" + pressure_data_name

        angles_data_np = load_angles_df_as_np(angles_data_path)
        pressure_data_np = load_df_as_np(pressure_data_path)

        noised_pressure_data_np = add_noise_on(pressure_data_np.copy(), low=200, high=500)

        angles_data_np_normalized = normalize_angle_data(angles_data_np, angle_idx=4)
        pressure_data_np_normalized, mean_pressure, std_pressure = normalize_pressure_data(pressure_data_np.copy(), pressure_idx=1)
        # pressure_data_np_normalized, mean_pressure, std_pressure = normalize_pressure_data(noised_pressure_data_np.copy(),
        #                                                                                    pressure_idx=1)

        pressure_distribution = get_pressure_data_distribution(pressure_data_np.copy())

        dataset = align_sensor_camera(camera_data=angles_data_np_normalized, sensor_data=pressure_data_np_normalized)

        dataset = pd.DataFrame(dataset)
        print(f"Raw pressure data: {pressure_data_np.shape[0]}, Raw angles data: {angles_data_np.shape[0]}, Dataset: {dataset.shape[0]}")
        if whether_save_every_dataset:
            dataset.to_csv(dir + session + "/dataset.csv", index=False)
            save_mean_and_std(mean_pressure, std_pressure, save_path=dir + session + "/pressure_mean_std.txt")

        if session in merge_list:
            total_dataset = pd.concat([total_dataset, dataset], axis=0, ignore_index=True)
        mean_pressure_list.append(mean_pressure)
        std_pressure_list.append(std_pressure)

    # mean = np.mean(mean_pressure_list, axis=0)
    # std = np.mean(std_pressure_list, axis=0)

    mean = np.array(np.array(mean_pressure_list).mean())
    std = np.array(np.array(std_pressure_list).std())

    return total_dataset, mean, std

        
    
    

if __name__ == "__main__":

    record_dir = "record"
    session_dir = record_dir + "/"
    #
    session_list = [
        # "session_20260314_195645",
        #             "session_20260316_133259",
        #             "session_20260316_133505",
        #             "session_20260317_153225",
        #             "session_20260327_161819",
        #             "session_20260327_170436",
        #             "session_20260327_170706",
                    "session_20260413_205209",
                    "session_20260413_210622",
                    "session_20260413_210818",
                    "session_20260413_215453",
                    "session_20260413_215617",
                    "session_20260413_215820",
                    "session_20260413_215937",
                    "session_20260413_220616",
                    "session_20260413_220819",
                    "session_20260413_221048"
                    ]

    merge_list = [
        # "session_20260314_195645",
        # "session_20260316_133259",
        # "session_20260316_133505",
        # "session_20260317_153225",
        # "session_20260327_161819",
        # "session_20260327_170436",
        # "session_20260327_170706"
        "session_20260413_205209",
        "session_20260413_210622",
        "session_20260413_210818",
        "session_20260413_215453",
        "session_20260413_215617",
        "session_20260413_215820",
        "session_20260413_215937",
        "session_20260413_215937",
        # "session_20260413_220616",
        "session_20260413_220819",
        "session_20260413_221048"
    ]
    
    total_dataset, mean_pressure, std_pressure = merge_all_data_as_dataset(session_dir, session_list, merge_list,
                                                                           whether_save_every_dataset=True)

    pressure_data = total_dataset["pressure"].values
    angles_data = total_dataset["angles"].values

    # print(pressure_data.shape)
    # print(angles_data.shape)
    #
    # print(mean_pressure.shape)
    # print(std_pressure.shape)



    total_dataset.to_csv(record_dir + "/total_dataset.csv", index=False)

    save_mean_and_std(mean_pressure, std_pressure, save_path=record_dir + "/pressure_mean_std.txt")

    # print(angles_data_np.shape)
    # print(pressure_data_np.shape)
    print(total_dataset.shape)























