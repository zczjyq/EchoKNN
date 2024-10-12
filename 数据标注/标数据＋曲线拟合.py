import numpy as np
import struct
import matplotlib.pyplot as plt
import os  # 用于操作文件
from matplotlib.backend_bases import MouseButton  # 用于捕捉鼠标事件

op = int(input("输入要进行的操作："))
if op == 1:
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


    # Read and reshape image data
    file_path = r"D:\DevelopCode\pytorch\sonar\KNN\data\go22-26m.dat"
    img_8bit_matrix = read_sonar_data(file_path)
    ans = reshape_img_matrix(img_8bit_matrix)  # ans为最后读出来的三维数组

    # 存储鼠标经过点的坐标
    dragged_points = []


    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))


    def plot_sonar_slice(ans, k):
        slice_data = ans[:, :, k]
        normalized_data = normalize_data(slice_data)
        plt.imshow(normalized_data, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label='Echo Intensity (Normalized)')
        plt.title(f'Sonar Echo Intensity for Slice {k} (Normalized)')
        plt.xlabel('Column Index (j)')
        plt.ylabel('Row Index (i)')


    def on_mouse_press(event):
        """鼠标按下时触发，初始化坐标记录"""
        global dragged_points
        if event.button is MouseButton.LEFT:
            dragged_points = []  # 清空之前的坐标记录
            print(f"Mouse pressed at ({event.xdata}, {event.ydata})")


    def on_mouse_motion(event):
        """鼠标拖动时触发，记录拖动过程中的坐标"""
        if event.xdata is not None and event.ydata is not None:
            # 记录鼠标当前经过的点
            dragged_points.append((event.xdata, event.ydata))
            print(f"Mouse moved to ({event.xdata}, {event.ydata})")


    def on_mouse_release(event):
        """鼠标释放时触发，保存坐标到文件"""
        if event.button is MouseButton.LEFT:
            print("Mouse released. Saving points to file...")
            save_coordinates_to_file(dragged_points)


    def save_coordinates_to_file(points):
        """将记录的坐标保存到桌面的 txt 文件"""
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")  # 获取桌面路径
        file_path = os.path.join(desktop_path, "dragged_coordinates.txt")  # 文件路径
        with open(file_path, 'w') as f:
            for point in points:
                f.write(f"{point[0]}, {point[1]}\n")  # 将每个点写入文件
        print(f"Coordinates saved to {file_path}")


    # 绑定鼠标事件到图像
    fig, ax = plt.subplots()

    # 将鼠标事件连接到图像窗口
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    # 连续播放 sonar 图像并允许鼠标拖动
    plt.ion()

    for i in range(0, 500):
        plt.clf()  # 清除当前图像
        plot_sonar_slice(ans, i)  # 绘制第 i 个切片
        plt.pause(100)  # 暂停 0.1 秒，实现连续播放
elif op == 2:
    # 读取文件并获取坐标数据
    def read_coordinates(file_path):
        x_coords = []
        y_coords = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.split(', '))  # 读取每行中的两个数字
                if x_coords and x - x_coords[-1] > 10:
                    continue
                x_coords.append(x)
                y_coords.append(600 - y)
        return np.array(x_coords), np.array(y_coords)


    # 读取txt文件中的坐标数据
    file_path = r"C:\Users\Lenovo\Desktop\dragged_coordinates.txt"  # 替换为你的txt文件路径
    x_data, y_data = read_coordinates(file_path)

    # 进行高次多项式拟合（可以调整degree值来改变多项式阶数）
    degree = 4  # 指定多项式的阶数（例如4次多项式）

    coefficients = np.polyfit(x_data, y_data, degree)

    # 将系数转换为多项式方程
    polynomial = np.poly1d(coefficients)

    # 打印多项式方程
    print(f"拟合的 {degree} 次多项式方程为：\n{polynomial}")

    # 可视化原始点和拟合曲线
    plt.scatter(x_data, y_data, color='red', label='Data Points')  # 原始点
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = polynomial(x_fit)
    plt.plot(x_fit, y_fit, color='blue', label=f'Fitted {degree} Degree Polynomial')  # 拟合曲线

    # 设置图形属性
    plt.title(f'{degree} Degree Polynomial Fitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # 使X和Y轴的单位长度一致
    plt.gca().set_aspect('equal', adjustable='box')  # 确保XY轴比例一致

    plt.grid(True)
    plt.show()
