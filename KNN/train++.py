import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import ipywidgets as widgets
from IPython.display import display
from sklearn.neighbors import NearestNeighbors
import time
import train_functions


# ---------------深度卷积神经网络模型--------------
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
    
# ---------------模型训练前的准备---------------
train_num = 50 # 参与训练的图片数量

# Read and reshape image data
file_path = "./data/HEU0812_20240202_133556.dat"
img_8bit_matrix = train_functions.read_sonar_data(file_path, train_num)
ans = train_functions.reshape_img_matrix(img_8bit_matrix, train_num)  # ans为最后读出来的三维数组

# 准备测试数据和数据标签，show_labels控制是否显示标签，weather_plot控制是否绘制数据的可视化图表
ans_tensor, labels_tensor = train_functions.prepare_data_and_labels(
    ans, train_num, alpha_start=1, alpha_end = 1.1, show_labels=False, weather_plot=False
) # 运行到这里，会输出torch.Size([50, 1, 600, 512])

# ---------------预加载练过的模型---------------
# 初始化卷积网络
conv_net = DeepConvNet()
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练模型的权重（路径替换为你的模型路径）
conv_net.load_state_dict(torch.load("./model/SuperDeep2024-11-30_10-36-00.pth", map_location=device))
# 将模型移动到 GPU
conv_net.to(device)
# 将数据和标签移动到 GPU
ans_tensor = ans_tensor.to(device)
labels_tensor = labels_tensor.to(device)

# ---------------开始训练卷积神经网络---------------
train_functions.train_supervised_model(conv_net, ans_tensor, train_num = 6, labels= labels_tensor, batch_size=8, learning_rate=0.005)

# ---------------保存模型参数或者模型---------------
localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
torch.save(conv_net.state_dict(), f"./model/SuperDeep{localtime}.pth")
# torch.save(conv_net, f"./model/SuperDeep{localtime}.pth")  # 保存模型