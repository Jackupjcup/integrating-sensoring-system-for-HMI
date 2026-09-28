def read_txt_first_line_to_list(txt_path):
    """
    读取TXT文件第一行，转换为数字列表（处理空格/制表符分隔）
    :param txt_path: TXT文件路径
    :return: 第一行的数字列表，读取失败返回空列表
    """
    try:
        # 1. 打开文件（指定编码，避免中文/特殊字符报错）
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 2. 读取第一行并去除首尾空白（换行/空格/制表符）
            first_line = f.readline().strip()
            print(first_line)

            # 3. 空行处理
            if not first_line:
                print(f"⚠️ TXT文件第一行为空：{txt_path}")
                return []

            # 4. 按空格/制表符分割内容，转换为数字（int/float自动识别）
            # split()无参数时，自动分割所有空白（空格/制表符/多个空格）
            str_list = first_line.split()
            num_list = []
            for s in str_list:
                try:
                    # 优先转int，失败则转float
                    num = int(s)
                except ValueError:
                    num = float(s)
                num_list.append(num)

            return num_list

    except FileNotFoundError:
        print(f"❌ 未找到文件：{txt_path}")
        return []
    except Exception as e:
        print(f"❌ 读取文件出错：{e}")
        return []

def rotate_point_90_clockwise(x, y):
    """
    对单个坐标点 (x,y) 进行顺时针90度旋转（适配OpenCV图片坐标体系）
    :param x: 原始x坐标（像素值，整数）
    :param y: 原始y坐标（像素值，整数）
    :param img_width: 原始图片宽度（像素）
    :param img_height: 原始图片高度（像素）
    :return: 旋转后的新坐标 (new_x, new_y)
    """
    # 顺时针旋转90度的核心公式（适配OpenCV坐标）
    # new_x = y
    # new_y = img_width - x - 1
    new_y = x
    new_x = 1 - y
    return new_x, new_y

def rotate_all_points(values):

    points = values[5:]

    num_points = 21

    for i in range(num_points):
        new_x, new_y = rotate_point_90_clockwise(points[3*i], points[3*i+1])
        points[3 * i] = new_x
        points[3 * i + 1] = new_y

    return values[: 5] + points

















