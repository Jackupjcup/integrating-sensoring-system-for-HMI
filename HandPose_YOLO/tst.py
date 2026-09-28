import cv2
import numpy as np
from yolo_utils import read_txt_first_line_to_list, rotate_all_points

# ===================== 你只需要改这里 =====================
image_path = "img.png"  # 你的图片
rotation_degree = cv2.ROTATE_90_CLOCKWISE
output_path = "hand_with_keypoints.jpg"
# ==========================================================


# 解析标签
values = read_txt_first_line_to_list("IMG_00000001.txt")

values = rotate_all_points(values)
print(values)

cls_id = values[0]
points = np.array(values[5:]).reshape(21, 3)  # (21,3)

# 读取图片
img = cv2.imread(image_path)
img = cv2.rotate(img, rotation_degree)
h, w = img.shape[:2]


# 画21个关键点
for i, (nx, ny, v) in enumerate(points):
    x = int(nx * w)
    y = int(ny * h)
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

# 手的骨架连线（标准21点连线）
skeleton = [
    (0,1),(1,2),(2,3),(3,4),       # 拇指
    (0,5),(5,6),(6,7),(7,8),       # 食指
    (0,9),(9,10),(10,11),(11,12),  # 中指
    (0,13),(13,14),(14,15),(15,16),# 无名指
    (0,17),(17,18),(18,19),(19,20) # 小指
]

for a,b in skeleton:
    nx1, ny1, v1 = points[a]
    nx2, ny2, v2 = points[b]
    x1 = int(nx1*w)
    y1 = int(ny1*h)
    x2 = int(nx2*w)
    y2 = int(ny2*h)
    cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 2)

# 保存
cv2.imwrite(output_path, img)
print(f"✅ 已生成带关键点的图片：{output_path}")