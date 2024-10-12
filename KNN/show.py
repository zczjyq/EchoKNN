import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import ipywidgets as widgets
from IPython.display import display
from sklearn.neighbors import NearestNeighbors


def read_sonar_data(file_path):
    # Read sonar beam data from a .dat file and return a matrix of image data
    valid_ping_num = 500  # Number of valid pings
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

def reshape_img_matrix(img_8bit_matrix):
    # Ensure img_8bit_matrix has the shape (307200, 500)
    if img_8bit_matrix.shape != (307200, 500):
        raise ValueError("img_8bit_matrix must have shape (307200, 500)")

    # Reshape each column into a (600, 512) array
    reshaped_arrays = np.empty((600, 512, img_8bit_matrix.shape[1]), dtype=np.uint8)

    for k in range(img_8bit_matrix.shape[1]):
        reshaped_arrays[:, :, k] = img_8bit_matrix[:, k].reshape((600, 512))

    return reshaped_arrays

def plot_anomalies(original_data, predictions, slice_idx):
    original_img = original_data[slice_idx, 0, :, :].cpu().numpy()
    prediction_mask = predictions[slice_idx, 0, :, :].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original Image (Slice {slice_idx})")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(original_img, cmap='gray')
    plt.imshow(prediction_mask, cmap='Reds', alpha=0.5)
    plt.title(f"Predicted Anomalies (Slice {slice_idx})")
    plt.colorbar()
    plt.show()


# 7. 加载新数据并运行检测
if __name__ == "__main__":
    # 假设有新的声呐数据（替换为实际的数据文件路径）
    file_path_test = "./data/go15-22m.dat"
    img_8bit_matrix_test = read_sonar_data(file_path_test)
    test = reshape_img_matrix(img_8bit_matrix_test)  # ans为最后读出来的三维数组

    # Read and reshape image data
    file_path = "./data/go22-26m.dat"
    img_8bit_matrix = read_sonar_data(file_path)
    ans = reshape_img_matrix(img_8bit_matrix)  # ans为最后读出来的三维数组
    # ans.shape: (307200, 500)
    cha = ans[:, :, 0] - test[:, :, 0]

    # 绘制差异矩阵的第一组数据
    plt.figure(figsize=(10, 5))

    # 显示第一张差异数据
    plt.subplot(1, 3, 1)
    plt.imshow(cha, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(ans[:, :, 0], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(test[:, :, 0], cmap='gray')
    plt.title('Difference Matrix - First Slice')
    plt.colorbar(label='Difference Intensity')

    plt.show()