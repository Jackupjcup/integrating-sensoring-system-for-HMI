if pressure_data.shape[0] == 10:
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle("Batch Palm Pressure Maps (2×5)", fontsize=16, y=0.98)

    axes_flat = axes.flatten()





def plot_pressure_heat(pressure_data, plot_name, whether_square_shape=False):
    # init canvas
    fig, ax = plt.subplots(figsize=(10, 10))
    # initial_data = np.zeros(TOTAL_NUM)
    plt.title(plot_name + " Palm Pressure Map")

    if whether_square_shape:
        im = ax.imshow(pressure_data.reshape(ROWS, COLS), cmap="coolwarm", aspect="auto", vmin=0, vmax=MAX_VALUE)
        plt.colorbar(im, label='Pressure Intensity')

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
        plt.show()
    else:
        scatter = ax.scatter(
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

        # 添加侧边颜色条
        plt.colorbar(scatter, label='Pressure Intensity')

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

        plt.show()