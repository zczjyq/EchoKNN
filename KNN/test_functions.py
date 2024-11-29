import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

# 深度卷积神经网络模型
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        # 编码器部分
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )

        # 新增一个更深的编码器分支
        self.encoder3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
                # 新增一个更深的编码器分支
        self.encoder4 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=25, stride=1, padding=12),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )

        # 解码器部分
        # 解码器部分，调整卷积转置层的参数来减少输出尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=2, stride=2),  # 将通道从 384 减少到 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2),  # 减少通道数，增加空间分辨率
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 减少输出到目标大小
            nn.Sigmoid(),
        )


    def forward(self, x):
        # 并行编码
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(x)
        encoded3 = self.encoder3(x)
        encoded4 = self.encoder4(x)

        # 将三个编码器的输出拼接在一起
        encoded = torch.cat((encoded1, encoded2, encoded3, encoded4), dim=1)  # 在通道维度拼接

        # 解码过程
        decoded = self.decoder(encoded)
        return decoded
    

# 1. 加载训练好的模型
def load_trained_model(model_path): # 直接加载 TorchScript 模型
    model = torch.jit.load(model_path)
    return model

def load_original_trained_model(model_path): # 加载模型参数
    model = DeepConvNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

# 2. 预处理新数据
def preprocess_new_data(new_sonar_data):
    # 检查数据维度，如果是2D则增加批量和通道维度
    if new_sonar_data.ndim == 2:
        new_sonar_data = np.expand_dims(new_sonar_data, axis=(0, 1))  # 转换为 (1, 1, height, width)
    elif new_sonar_data.ndim != 4:
        raise ValueError("Input data must be a 2D array or a 4D tensor")

    # 归一化数据
    new_sonar_data = (new_sonar_data - np.min(new_sonar_data)) / (np.max(new_sonar_data) - np.min(new_sonar_data))
    new_data_tensor = torch.tensor(new_sonar_data, dtype=torch.float32)

    return new_data_tensor


# 3. 使用模型进行推断
def detect_anomalies(model, data_tensor):
    with torch.no_grad():  # 禁用梯度计算以加快推断速度
        predictions = model(data_tensor)
    return predictions


# 4. 可视化原始数据与异常点
def plot_anomalies(original_data, predictions, slice_idx, photo_show):
    original_img = original_data[slice_idx, 0, :, :].cpu().numpy()  # 获取原始数据
    prediction_mask = predictions[slice_idx, 0, :, :].cpu().numpy()  # 获取模型预测的异常点

    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original Image (Slice {photo_show})")
    plt.colorbar()

    # 显示预测的异常点
    plt.subplot(1, 2, 2)
    plt.imshow(original_img, cmap='gray')  # 灰度图作为背景
    plt.imshow(prediction_mask, cmap='Reds', alpha=0.5)  # 红色的异常点叠加显示
    plt.title(f"Predicted Anomalies (Slice {photo_show})")
    plt.colorbar()

    plt.show()


# 5. 综合步骤：加载模型、处理新数据、进行预测并可视化
def run_anomaly_detection(model_path, new_sonar_data, photo_show):
    # 尝试加载模型
    try:
        # 首先尝试加载 TorchScript 模型
        model = load_trained_model(model_path)
    except Exception as e:
        try:
            # 如果失败，尝试加载模型参数
            model = load_original_trained_model(model_path)
        except Exception as e:
            raise Exception(f"无法加载模型，错误信息: {str(e)}")

    # 预处理新数据
    data_tensor = preprocess_new_data(new_sonar_data)

    # 使用模型进行推断，检测异常点
    predictions = detect_anomalies(model, data_tensor)

    # 可视化异常点
    plot_anomalies(data_tensor, predictions, 0, photo_show)


# 6. 加载图片并转换为模型所需的格式
def load_image_as_tensor(image_path, target_size=(512, 600)):
    """
    读取图片并将其转换为模型所需的格式（600x512的张量）。
    """
    # 读取图片并转换为灰度图
    image = Image.open(image_path).convert("L")  # 灰度模式

    # 调整图片大小为目标大小
    image_resized = image.resize(target_size, Image.ANTIALIAS)

    # 转换为 numpy 数组
    image_array = np.array(image_resized, dtype=np.uint8)

    # 归一化为 [0, 1] 范围
    image_normalized = image_array / 255.0

    # 转换为 PyTorch 张量，形状为 (1, 1, 600, 512)
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor


# 7. 直接输入图像张量，利用模型进行异常点检测。
def run_anomaly_detection_for_image(model_path, image_tensor, photo_show):
    """
    直接输入图像张量，利用模型进行异常点检测。
    """
    # 加载训练好的模型
    model = load_trained_model(model_path)

    # 确保输入张量形状为 (1, 1, 600, 512)
    if image_tensor.shape != (1, 1, 600, 512):
        raise ValueError(f"Input tensor shape must be (1, 1, 600, 512), got {image_tensor.shape}")

    # 使用模型进行推断，检测异常点
    predictions = detect_anomalies(model, image_tensor)

    # 可视化结果
    plot_anomalies(image_tensor, predictions, 0,  photo_show)
