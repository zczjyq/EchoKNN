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

# --------------------------------------------------------------------

# 两条墙的四次多项式方程
def right_wall(x):
    # 右侧墙的四次方程
     return 2.408e-07 *x**4 - 0.0004357 *x**3 + 0.2981 *x**2 - 91.88 *x + 1.099e+04


def left_wall(x):
    # 左侧墙的四次方程
    return 7.342e-07 * x ** 4 - 0.000102 * x ** 3 + 0.01065 * x ** 2 + 0.4626 * x + 184




def read_sonar_data(file_path, train_num):
    # Read sonar beam data from a .dat file and return a matrix of image data
    valid_ping_num = train_num  # Number of valid pings
    num_IMG = 307200  # Total image data points per ping

    img_8bit_matrix = np.zeros(
        (num_IMG, valid_ping_num), dtype=np.uint8
    )  # Initialize the output matrix

    with open(file_path, "rb") as file:
        file.read(40)  # Read and discard the file header (10 uint32)

        # Fixed header search
        c1 = struct.unpack("B", file.read(1))[0]
        c2 = struct.unpack("B", file.read(1))[0]
        c3 = struct.unpack("B", file.read(1))[0]
        for k in range(valid_ping_num):
            if file.read(0) == "b":
                break
            # Header search within 64k
            for kk in range(65535):
                c4 = struct.unpack("B", file.read(1))[0]
                if c4 == ord("O") and c3 == ord("S") and c2 == 0 and c1 == 0:
                    break
                c1, c2, c3 = c2, c3, c4

                if file.read(0) == "b":
                    return img_8bit_matrix

            if kk == 65535:
                return img_8bit_matrix
            file.read(14)  # Read and discard SHeaderStr

            mode = np.frombuffer(file.read(5), dtype=np.uint8)
            range_percent = struct.unpack("d", file.read(8))[0]
            gain_percent = struct.unpack("d", file.read(8))[0]
            speed_of_sound = struct.unpack("d", file.read(8))[0]
            salinity = struct.unpack("d", file.read(8))[0]
            ext_flags = struct.unpack("I", file.read(4))[0]
            reserved = np.frombuffer(file.read(32), dtype=np.uint32)

            num_dat16 = reserved[4]  # Sample points
            chn_dat16 = reserved[5]  # Channels

            ping_id = struct.unpack("I", file.read(4))[0]
            status = np.frombuffer(file.read(69), dtype=np.uint8)
            range_resolution = struct.unpack("d", file.read(8))[0]
            n_ranges = struct.unpack("H", file.read(2))[0]
            n_beams = struct.unpack("H", file.read(2))[0]
            file.read(28)  # Read and discard spare

            # Update sample points and beams
            num_sam = n_ranges
            num_beam = float(n_beams)

            file.read(1024)  # Read and discard beam angles (512 int16)
            file.read(822)  # Read and discard tem16 (411 uint16)

            # Read image data
            img_8bit = np.frombuffer(file.read(num_IMG), dtype=np.uint8)
            img_8bit_matrix[:, k] = img_8bit

    return img_8bit_matrix


def reshape_img_matrix(img_8bit_matrix, train_num):
    # Ensure img_8bit_matrix has the shape (307200, 500)
    if img_8bit_matrix.shape != (307200, train_num):
        raise ValueError("img_8bit_matrix must have shape (307200, 500)")

    # Reshape each column into a (600, 512) array
    reshaped_arrays = np.empty((600, 512, img_8bit_matrix.shape[1]), dtype=np.uint8)

    for k in range(img_8bit_matrix.shape[1]):
        reshaped_arrays[:, :, k] = img_8bit_matrix[:, k].reshape((600, 512))

    return reshaped_arrays


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_sonar_slice(ans, k):
    slice_data = ans[:, :, k]
    normalized_data = normalize_data(slice_data)
    plt.imshow(normalized_data, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(label="Echo Intensity (Normalized)")
    plt.title(f"Sonar Echo Intensity for Slice {k} (Normalized)")
    plt.xlabel("Column Index (j)")
    plt.ylabel("Row Index (i)")


# 生成标签，墙体附近的点设为异常，labels为二维矩阵
def generate_labels(data_shape, threshold=5):
    labels = torch.zeros(data_shape[0], data_shape[1])  # 只需要二维的标签矩阵

    for x in range(data_shape[1]):
        y1 = int(right_wall(x))
        y2 = int(left_wall(x))
        for i in range(max(0, x - threshold), min(data_shape[1], x + threshold)):
            for j in range(max(0, y1 - threshold), min(data_shape[0], y1 + threshold)):
                labels[j, i] = 1
            for j in range(max(0, y2 - threshold), min(data_shape[0], y2 + threshold)):
                labels[j, i] = 1

    return labels


# 可视化结果
def plot_anomalies(original_data, predictions, slice_idx):
    original_img = original_data[slice_idx, 0, :, :].cpu().numpy()
    prediction_mask = predictions[slice_idx, 0, :, :].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title(f"Original Image (Slice {slice_idx})")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(original_img, cmap="gray")
    plt.imshow(prediction_mask, cmap="Reds", alpha=0.5)
    plt.title(f"Predicted Anomalies (Slice {slice_idx})")
    plt.colorbar()
    plt.show()


# 添加交互式功能
def interactive_plot(ans_tensor, predictions):
    def update_plot(slice_idx):
        plot_anomalies(ans_tensor, predictions, slice_idx)

    slice_slider = widgets.IntSlider(
        value=0, min=0, max=20 - 1, step=1, description="Slice:"
    )
    widgets.interact(update_plot, slice_idx=slice_slider)
# 绘制叠加的原始数据和标签
def plot_data_with_labels(data, labels, show_labels=True):
    plt.figure(figsize=(8, 6))

    # 绘制原始数据
    plt.imshow(data, cmap="gray", interpolation="none")
    if show_labels:
    # 绘制标签，使用半透明效果
        plt.imshow(labels, cmap="autumn", interpolation="none", alpha=0.5)

    plt.colorbar(label="Anomaly Label")
    plt.title("Original Data with Anomaly Labels")
    plt.xlabel("X-axis (Image Width)")
    plt.ylabel("Y-axis (Image Height)")
    plt.show()

def enhance_contrast(data, alpha=2):
    """
    增强数据的对比度，使得数值小的更加小，数值大的更加大。
    
    参数:
    - data: 输入数据，要求已经归一化在[0, 1]之间。
    - alpha: 控制增强程度的参数，alpha > 1 时，增强效果更强。
    
    返回值:
    - enhanced_data: 对比度增强后的数据，范围仍然在[0, 1]之间。
    """
    # 使用幂函数进行数据的非线性变换，alpha 控制变换的强度
    enhanced_data = np.power(data, alpha)
    
    return enhanced_data


# 训练卷积神经网络，使用二元交叉熵损失
def train_supervised_model(model, data, train_num, labels, batch_size=16, learning_rate=0.001):
    num_epochs= train_num
    print("Training supervised model...")
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_loader = torch.utils.data.DataLoader(list(zip(data, labels)), batch_size=batch_size, shuffle=True)
    t = time.time()
    for epoch in range(num_epochs):
        for batch_data, batch_labels in data_loader:
            output = model(batch_data)
            loss = criterion(output, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 清空缓存显存
        torch.cuda.empty_cache()
        
        t1 = time.time()
        if ((epoch + 1) % 10 == 0) or t1 - t > 5:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {t1 - t}')
            t = t1



# 准备数据和标签
def prepare_data_and_labels(ans, train_num, alpha_start=0.5, alpha_end=1.5, alpha_step=0.2, show_labels = True, weather_plot = True, show_all = True):
    data = np.moveaxis(ans, -1, 0)
    data = np.expand_dims(data, axis=1)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    alpha = np.arange(alpha_start, alpha_end, alpha_step)
    res =enhance_contrast(data, alpha=0.4)
    
    enhanced_data_list = []

    for i in alpha:
        res = enhance_contrast(data, alpha=i)
        enhanced_data_list.append(res)
    res = np.concatenate(enhanced_data_list, axis=0)

    data_tensor = torch.tensor(res, dtype=torch.float32)
    print(data_tensor.shape)
    # 生成二维标签矩阵
    labels_tensor = generate_labels(data_tensor.shape[2:])
    if weather_plot:
    # 绘制叠加的原始数据和标签
        if show_all:
            for i in range(0, data_tensor.shape[0], train_num):
                plot_data_with_labels(data_tensor[i, 0].numpy(), labels_tensor.numpy(), show_labels)

    # 将二维标签扩展为与数据相同的三维尺寸
    labels_tensor = labels_tensor.unsqueeze(0).expand(data_tensor.shape[0], -1, -1)
    labels_tensor = labels_tensor.unsqueeze(1)

    return data_tensor, labels_tensor



def read_sonar_data(file_path, nums_photo):
    # Read sonar beam data from a .dat file and return a matrix of image data
    valid_ping_num = nums_photo  # Number of valid pings
    num_IMG = 307200  # Total image data points per ping

    img_8bit_matrix = np.zeros((num_IMG, valid_ping_num), dtype=np.uint8)  # Initialize the output matrix

    with open(file_path, 'rb') as file:
        file.read(40)  # Read and discard the file header (10 uint32)

        # Fixed header search
        c1 = struct.unpack('B', file.read(1))[0]
        c2 = struct.unpack('B', file.read(1))[0]
        c3 = struct.unpack('B', file.read(1))[0]
        for k in range(valid_ping_num):
            if file.read(0) == 'b':
                break
            # Header search within 64k
            for kk in range(65535):
                c4 = struct.unpack('B', file.read(1))[0]
                if c4 == ord('O') and c3 == ord('S') and c2 == 0 and c1 == 0:
                    break
                c1, c2, c3 = c2, c3, c4

                if file.read(0) == 'b':
                    return img_8bit_matrix

            if kk == 65535:
                return img_8bit_matrix
            file.read(14)  # Read and discard SHeaderStr

            mode = np.frombuffer(file.read(5), dtype=np.uint8)
            range_percent = struct.unpack('d', file.read(8))[0]
            gain_percent = struct.unpack('d', file.read(8))[0]
            speed_of_sound = struct.unpack('d', file.read(8))[0]
            salinity = struct.unpack('d', file.read(8))[0]
            ext_flags = struct.unpack('I', file.read(4))[0]
            reserved = np.frombuffer(file.read(32), dtype=np.uint32)

            num_dat16 = reserved[4]  # Sample points
            chn_dat16 = reserved[5]  # Channels

            ping_id = struct.unpack('I', file.read(4))[0]
            status = np.frombuffer(file.read(69), dtype=np.uint8)
            range_resolution = struct.unpack('d', file.read(8))[0]
            n_ranges = struct.unpack('H', file.read(2))[0]
            n_beams = struct.unpack('H', file.read(2))[0]
            file.read(28)  # Read and discard spare

            # Update sample points and beams
            num_sam = n_ranges
            num_beam = float(n_beams)

            file.read(1024)  # Read and discard beam angles (512 int16)
            file.read(822)  # Read and discard tem16 (411 uint16)

            # Read image data
            img_8bit = np.frombuffer(file.read(num_IMG), dtype=np.uint8)
            img_8bit_matrix[:, k] = img_8bit

    return img_8bit_matrix


def reshape_img_matrix(img_8bit_matrix, nums_photo):
    # Ensure img_8bit_matrix has the shape (307200, 500)
    if img_8bit_matrix.shape != (307200, nums_photo):
        raise ValueError("img_8bit_matrix must have shape (307200, 500)")

    # Reshape each column into a (600, 512) array
    reshaped_arrays = np.empty((600, 512, img_8bit_matrix.shape[1]), dtype=np.uint8)

    for k in range(img_8bit_matrix.shape[1]):
        reshaped_arrays[:, :, k] = img_8bit_matrix[:, k].reshape((600, 512))

    return reshaped_arrays