import torch.nn as nn
import torch.optim as optim
import torch
from spatial_mask_param import SpatialMask

# class SmallCNN_regression(nn.Module):
#     def __init__(self, num_classes=14):
#         super(SmallCNN_regression, self).__init__()
#         # 特征提取层：适配14×10的输入尺寸
#         self.features = nn.Sequential(
#             # 卷积层1：输入通道1，输出通道16，卷积核3×3，padding=1（保证尺寸不缩小）
#             nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
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
            nn.Conv2d(5, 16, 3, padding=1),
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
                # nn.Dropout(0.5),
                nn.Linear(64, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        x = self.features(x)  # [B, 32, 3, 2]
        x = x.view(x.size(0), -1)

        outputs = []
        for head in self.heads:
            outputs.append(head(x))

        # 拼成 (batch,14)
        return torch.cat(outputs, dim=1)


'''------------ Multi-Mask CNN -----------'''

# class SmallCNN_regression(nn.Module):
#     def __init__(self, num_classes=14):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.spatial_mask_params = SpatialMask(device=self.device)
#
#         # 14个独立head
#         self.multi_features = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(1, 16, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),  # 7×5
#
#                 nn.Conv2d(16, 32, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)  # 3×2
#             ) for _ in range(num_classes)
#         ])
#
#         self.flatten_dim = 32 * 3 * 2
#
#         self.multi_regressor = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.flatten_dim, 64),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(64, 1)
#             ) for _ in range(num_classes)
#         ])
#
#     def forward(self, x):
#
#         outputs = []
#         for i, feature_branch in enumerate(self.multi_features):
#             x = x * self.spatial_mask_params.masks_tensor[self.spatial_mask_params.mask_keys_list[i]]
#             features = feature_branch(x)
#
#             features_flat = features.view(x.size(0), -1)
#
#             outputs.append(self.multi_regressor[i](features_flat))
#
#         # 拼成 (batch,14)
#         return torch.cat(outputs, dim=1)







