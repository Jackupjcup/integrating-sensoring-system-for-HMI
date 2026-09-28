import os
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def load_pressure_data(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    csv_files.sort()

    df_list = []
    label_idx = folder_path[-1]

    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(folder_path, file), header=None)
        df["label"] = int(label_idx) - 1
        df_list.append(df)

    return df_list

def raw_gesture_to_cnn_img(hand_gesture_list, fp, whether_save_all=True):

    if os.path.exists(fp + "cnn_data.npz") and os.path.exists(fp + "Merged_Data.csv"):
        merged_df = pd.read_csv(fp + "merged_data.csv")
        loaded_data = np.load(fp + "cnn_data.npz")
        images_np = loaded_data["images"]
        labels_np = loaded_data["labels"]
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

        if whether_save_all:
            np.savez_compressed(
                fp + "cnn_data.npz",
                images=images_np,
                labels=labels_np,
                allow_pickle=False
            )
            merged_df.to_csv(fp + "merged_data.csv", index=False)
            print("图像+标签已保存为cnn_data.npz")

    return merged_df, images_np, labels_np


class CNNDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 转换为PyTorch张量
        image = torch.from_numpy(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SmallCNN, self).__init__()
        # 特征提取层：适配14×10的输入尺寸
        self.features = nn.Sequential(
            # 卷积层1：输入通道1，输出通道16，卷积核3×3，padding=1（保证尺寸不缩小）
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 池化层1：2×2池化，尺寸变为 (14/2)×(10/2) = 7×5
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层2：输入通道16，输出通道32，卷积核3×3，padding=1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 池化层2：2×2池化，尺寸变为 (7//2)×(5//2) = 3×2（向下取整）
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层：根据池化后的尺寸计算输入维度（32×3×2=192）
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 2, 64),  # 隐藏层64个神经元
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(64, num_classes)  # 输出类别数
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平为一维向量
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x


def evaluate_model(model, test_loader, device, epoch=None):
    """
    评估模型性能，返回准确率和所有真实标签/预测标签（用于混淆矩阵）
    :param model: 训练中的模型
    :param test_loader: 测试集加载器
    :param device: 设备（cuda/cpu）
    :param epoch: 当前epoch数（用于打印）
    :return: test_acc, all_labels, all_preds
    """
    model.eval()  # 切换评估模式
    correct = 0
    total = 0
    all_labels = []  # 存储所有真实标签
    all_preds = []  # 存储所有预测标签

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 累计总数和正确数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集所有标签和预测结果（用于混淆矩阵）
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算准确率
    test_acc = 100 * correct / total

    # 打印当前epoch的测试结果
    if epoch is not None:
        print(f"Epoch [{epoch + 1}] 测试准确率: {test_acc:.2f}%")
    else:
        print(f"最终测试准确率: {test_acc:.2f}%")

    return test_acc, np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(all_labels, all_preds, num_classes, save_path="confusion_matrix.png"):
    """
    生成并保存混淆矩阵
    :param all_labels: 所有真实标签
    :param all_preds: 所有预测标签
    :param num_classes: 类别数
    :param save_path: 混淆矩阵保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # 保存混淆矩阵
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")

def plot_train_curve(acc_list, loss_list):
    plt.figure(figsize=(10, 5))

    # 定义左Y轴：Loss
    ax1 = plt.gca()
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='red', fontsize=12)
    ax1.plot(loss_list, color='red', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='red')

    # 定义右Y轴：Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue', fontsize=12)
    ax2.plot(acc_list, color='blue', label='Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='blue')

    # 图例（两条线合并显示）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 标题
    plt.title('Training Loss & Accuracy Curve', fontsize=14)

    # 显示并保存
    plt.tight_layout()
    plt.savefig('train_curve.png', dpi=300)  # 高清保存
    plt.show()

def main(images_np, labels_np,
         model_save_path="trained_cnn_model.pth",
         cm_save_path="confusion_matrix.png",
         num_epochs=10):
    # 设备配置：优先使用GPU，没有则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数设置
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = num_epochs
    num_classes = len(np.unique(labels_np))
    train_ratio = 0.8
    model_save_path = model_save_path  # 模型保存路径
    cm_save_path = cm_save_path  # 混淆矩阵保存路径

    # 计算训练集样本数
    total_samples = len(images_np)
    train_size = int(total_samples * train_ratio)
    test_size = total_samples - train_size

    # 1. 加载数据集
    shuffle_idx = np.random.permutation(len(images_np))
    images_np = images_np[shuffle_idx]
    labels_np = labels_np[shuffle_idx]

    train_images = images_np[:train_size]
    train_labels = labels_np[:train_size]
    test_images = images_np[train_size:]
    test_labels = labels_np[train_size:]
    print(f"【数据信息】总样本数: {total_samples} | 训练集: {train_size} | 测试集: {test_size}")

    # Save dataset as train and test_on_dataset sets
    np.savez_compressed("train_set.npz", images=train_images, labels=train_labels)
    np.savez_compressed("test_set.npz", images=test_images, labels=test_labels)

    train_dataset = CNNDataset(train_images, train_labels)
    test_dataset = CNNDataset(test_images, test_labels)

    # 训练集：shuffle=True（每轮打乱），测试集：shuffle=False（无需打乱）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型、损失函数、优化器
    model = SmallCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    # Record training data
    accuracy_list = []
    train_loss_list = []
    # 3. 训练模型
    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移到指定设备
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数

            running_loss += loss.item()

        # 打印每个epoch的损失
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Test
        test_acc, _, _ = evaluate_model(model, test_loader, device, epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_acc,
                'num_classes': num_classes
            }, model_save_path)
            print(f"✅ 最佳模型已更新保存（准确率: {best_acc:.2f}%）")

        accuracy_list.append(test_acc)
        train_loss_list.append(epoch_loss)

    # 4. 测试模型
    final_acc, all_labels, all_preds = evaluate_model(model, test_loader, device)
    # Plot training curve of accuracy and training loss
    plot_train_curve(accuracy_list, train_loss_list)
    # 生成并保存混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, num_classes, cm_save_path)

    print(f"\n📌 训练完成！")
    print(f"最终模型保存路径: {model_save_path}")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"最终测试准确率: {final_acc:.2f}%")

    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Final Test Accuracy = {final_acc:.4f}\n")
        f.write(f"Best Test Accuracy  = {best_acc:.4f}\n")




def load_model(model_path, num_classes=5, device=None):
    """
    加载训练好的模型
    :param model_path: 模型文件路径（如"trained_cnn_model.pth"）
    :param num_classes: 类别数（需和训练时一致）
    :param device: 运行设备（cuda/cpu）
    :return: 加载好的模型（eval模式）
    """
    # 自动选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = SmallCNN(num_classes=num_classes).to(device)

    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)  # map_location自动适配设备
    model.load_state_dict(checkpoint['model_state_dict'])

    # 切换到评估模式（禁用Dropout/BatchNorm）
    model.eval()
    print(f"✅ 模型已从 {model_path} 加载完成，运行在 {device} 上")
    return model


def predict_single_image(model, image_np, device=None):
    """
    对单张14×10图像进行预测
    :param model: 加载好的模型
    :param image_np: 待预测图像数组（原始形状：(14,10) 或 (1,14,10) 或 (1,1,14,10)）
    :param device: 运行设备
    :return: 预测标签（int）、预测概率（各分类的置信度）
    """
    # 自动选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 关键：图像预处理（必须和训练时一致） =====================
    # 步骤1：统一形状为 (1, 1, 14, 10)（batch_size=1, channel=1, H=14, W=10）
    if image_np.shape == (14, 10):
        image_np = np.expand_dims(image_np, axis=(0, 1))  # (14,10) → (1,1,14,10)
    elif image_np.shape == (1, 14, 10):
        image_np = np.expand_dims(image_np, axis=0)  # (1,14,10) → (1,1,14,10)
    elif image_np.shape != (1, 1, 14, 10):
        raise ValueError(f"图像形状错误！需为(14,10)/(1,14,10)/(1,1,14,10)，当前为{image_np.shape}")

    # 步骤2：转换为float32（和训练时一致）
    image_np = image_np.astype(np.float32)

    # 步骤3：转为PyTorch张量并移到指定设备
    image_tensor = torch.from_numpy(image_np).to(device)

    # ===================== 预测推理 =====================
    with torch.no_grad():  # 禁用梯度计算，加速+省内存
        outputs = model(image_tensor)  # 输出形状：(1, num_classes)
        # 计算各分类的概率（softmax）
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        # 获取预测标签（概率最大的类别）
        predicted_label = torch.argmax(outputs, dim=1).cpu().item()

    return predicted_label, probabilities


if __name__ == "__main__":
    COUNT = 0
    ROWS, COLS = 14, 10
    TOTAL_NUM = ROWS * COLS
    MAX_VALUE = 1000

    '''Hyperparameters'''
    npz_data_path = "D:/blender_workplace/modules/Functions/objects_recognition/merged_data.npz"
    data = np.load(npz_data_path)
    images_np = data["images"]
    labels_np = data["labels"]

    model_save_path = "trained_cnn_model.pth"
    cm_save_path = "confusion_matrix.png"

    '''Main training process'''
    main(images_np, labels_np, model_save_path, cm_save_path, num_epochs=30)


    '''Load model to predict single image'''
    # tst_model_save_path = model_save_path
    #
    # cnn_model = load_model(tst_model_save_path, num_classes=11)
    # # import single image_np
    # # tst_npz_path = "D:/blender_workplace/modules/Functions/Hand_Gesture/20260115-21/" + "cnn_data.npz"
    # tst_npz_path = "test_set.npz"
    # tst_loaded_data = np.load(tst_npz_path)
    # tst_images_np = tst_loaded_data["images"]
    # tst_labels_np = tst_loaded_data["labels"]
    # print(tst_labels_np)
    #
    # tst_idx = 5
    # tst_label = tst_labels_np[tst_idx]
    # pred_label, probs = predict_single_image(cnn_model, tst_images_np[tst_idx])
    #
    # print(f"tst_label is {tst_label}\npredicted label is {pred_label}\n probs: {probs}")

















