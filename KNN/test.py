import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import CQNet
import test_functions
from PIL import Image

nums_photo = 500

# ----------------预测.dat文件----------------
# 替换为实际的数据文件路径
new_file_path = "KNN/data/HEU0812_20240202_133556.dat"

# 读取并处理声呐数据
img_8bit_matrix = CQNet.read_sonar_data(new_file_path, nums_photo)

# 模型文件路径
trained_model_path = 'KNN/model/SuperDeep_scripted.pt'
# trained_model_path = './model/SuperDeep.pth'

# 循环检测每张图片
for i in range(1, 2):
    photo_show = i # 显示第 i 张图片
    # 重新调整数据形状
    new_sonar_data = CQNet.reshape_img_matrix_single(img_8bit_matrix, photo_show)  # 重新调整数据形状
    # 运行异常点检测
    test_functions.run_anomaly_detection(trained_model_path, new_sonar_data, photo_show)
    
    
# ----------------预测图片----------------
# 定义图片路径
image_path = "KNN/picture/bike.png"  # 替换为您的图片路径

# 读取图片并转换为模型输入格式
image_tensor = test_functions.load_image_as_tensor(image_path)

# 模型路径
trained_model_path = 'KNN/model/SuperDeep_scripted.pt'

# 只处理一张图片
photo_show = 1

# 运行异常点检测
test_functions.run_anomaly_detection_for_image(trained_model_path, image_tensor, photo_show)