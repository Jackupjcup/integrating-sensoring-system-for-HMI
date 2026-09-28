import torch.nn as nn
import torch.optim as optim
import torch
# class SmallCNN_regression(nn.Module):
#     def __init__(self, num_classes=14):
#         super(SmallCNN_regression, self).__init__()
#         # 特征提取层：适配14×10的输入尺寸
#         self.features = nn.Sequential(
#             # 卷积层1：输入通道1，输出通道16，卷积核3×3，padding=1（保证尺寸不缩小）
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             # 池化层1：2×2池化，尺寸变为 (14/2)×(10/2) = 7×5
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # 卷积层2：输入通道16，输出通道32，卷积核3×3，padding=1
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             # 池化层2：2×2池化，尺寸变为 (7//2)×(5//2) = 3×2（向下取整）
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.regressor = nn.Sequential(
#             nn.Linear(32 * 3 * 2, 128),  # 可增大隐藏层（如128）适配多值输出
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)  # 输出14个连续值
#         )
#
#     def forward(self, x):
#         # 特征提取
#         x = self.features(x)
#         # 展平为一维向量
#         x = x.view(x.size(0), -1)
#         # 回归
#         x = self.regressor(x)
#         return x

'''------------ Multi Regressor Output -----------'''

class SmallCNN_regression(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 7×5

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 3×2
        )

        self.flatten_dim = 32 * 3 * 2

        # 14个独立head
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flatten_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        outputs = []
        for head in self.heads:
            outputs.append(head(x))

        # 拼成 (batch,14)
        return torch.cat(outputs, dim=1)

'''------------ *C*B*A*M* -----------'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#

#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio if in_planes > ratio else 1, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio if in_planes > ratio else 1, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):  # 针对14x10小尺寸，改用3x3卷积
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=3):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         x = x * self.ca(x)
#         x = x * self.sa(x)
#         return x
#
#
# # --- 2. 整合 CBAM 的增强版 SmallCNN ---
#
# class SmallCNN_regression(nn.Module):
#     def __init__(self, num_classes=14):
#         super(SmallCNN_regression, self).__init__()
#
#         # 第一层卷积块
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             CBAM(16),  # 在第一次池化前提取特征权重
#             nn.MaxPool2d(kernel_size=2, stride=2)  # 变为 7x5
#         )
#
#         # 第二层卷积块
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             CBAM(32),  # 在第二次池化前提取特征权重
#             nn.MaxPool2d(kernel_size=2, stride=2)  # 变为 3x2
#         )
#
#         # 回归层
#         self.regressor = nn.Sequential(
#             nn.Linear(32 * 3 * 2, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         # 通过增强的特征提取层
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#
#         # 展平
#         x = x.view(x.size(0), -1)
#
#         # 回归输出
#         x = self.regressor(x)
#         return x
















