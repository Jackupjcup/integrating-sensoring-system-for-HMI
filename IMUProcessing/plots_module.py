import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(imu_raw_data, whether_print_mean=True):
    if whether_print_mean:
        print(f'AccX: {np.mean(imu_raw_data.iloc[:, 0])}')
        print(f'AccY: {np.mean(imu_raw_data.iloc[:, 1])}')
        print(f'AccZ: {np.mean(imu_raw_data.iloc[:, 2])}')
        print()
        print(f'GyroX: {np.mean(imu_raw_data.iloc[:, 3])}')
        print(f'GyroY: {np.mean(imu_raw_data.iloc[:, 4])}')
        print(f'GyroZ: {np.mean(imu_raw_data.iloc[:, 5])}')

    row_num, col_num = 2, 3
    for i in range(1, len(imu_raw_data.columns) + 1):
        plt.subplot(row_num, col_num, i)
        n, bins, patches = plt.hist(x=imu_raw_data[imu_raw_data.columns[i - 1]],
                                    bins=50,
                                    color='blue',
                                    alpha=0.75,
                                    edgecolor='black')
        plt.title(imu_raw_data.columns[i - 1])
    plt.show()

def three_imu_raw_data_sequencs(file_path):
    # fig = plt.figure()
    for i in range(3):
        imu_data_dic = file_path + f'/imu_{i+1}/imu_animation_data.csv'
        imu_data = pd.read_csv(imu_data_dic)
        plt.subplot(3, 3, i*3 + 1)
        plt.plot(imu_data.iloc[:, 3], label="GyroX")
        plt.legend()
        plt.subplot(3, 3, i*3 + 2)
        plt.plot(imu_data.iloc[:, 4], label="GyroY")
        plt.legend()
        plt.subplot(3, 3, i*3 + 3)
        plt.plot(imu_data.iloc[:, 5], label="GyroZ")
        plt.legend()
    plt.show()


def acc_vector_gathering_plot(imu_raw_data):
    acc_np = imu_raw_data.iloc[:, :3].values

    gravity_np = np.sqrt(np.sum(acc_np**2, axis=1))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x0, y0, z0 = 0, 0, 0

    for i in range(len(acc_np)):
        ax.quiver(x0, y0, z0, acc_np[i][0], acc_np[i][1], acc_np[i][2],
          color='g', linewidth=2, arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('IMU Acc 3D Vectors')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    # plt.show()

    return gravity_np


if __name__ == '__main__':

    ### load the data
    file_path = r'../DataCSV/imu_3/raw_data.csv'
    # file_path = "DataCSV/imu_3/imu_animation_data.csv"
    imu_raw_data = pd.read_csv(file_path)

    ### functions to visualize the data
    # print(f' --- {file_path} --- \n')
    plot_histogram(imu_raw_data)
    # gravity_np = acc_vector_gathering_plot(imu_raw_data)


    ### compensate the bias of accelerator
    # imu_bias_data_np = pd.read_csv('DataCSV/IMUDataBias.csv').iloc[21: 27, :3].values
    # # print(imu_bias_data_np)
    # compensate_bias(imu_bias_data_np)

    # IMU raw data sequence
    # fp = "DataCSV"
    # three_imu_raw_data_sequencs(fp)








