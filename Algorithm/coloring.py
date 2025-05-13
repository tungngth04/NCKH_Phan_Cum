import numpy as np
import matplotlib.pyplot as plt
import rasterio
from Algorithm.FCM import FuzzyCMeans
import os
from Algorithm.CFCM import Dcfcm 

# Kiểm tra xem file có tồn tại không
file_path = 'E:/NCKH/data/LANDSAT/SCL/z50_B2_scl.tif'
if os.path.exists(file_path):
    print(f"File {file_path} tồn tại")
else:
    print(f"File {file_path} không tồn tại")
# Bước 1: Đường dẫn đến các ảnh vệ tinh .tif
image_paths1 = [
    'E:/NCKH/data/LANDSAT/SCL/z50_B2_scl.tif',
    # 'E:/NCKH/z50_B5_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B3_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B4_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B5_scl.tif',
]

image_paths2 = [
    'E:/NCKH/data/LANDSAT/TN/z50_B2_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z50_B3_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z50_B4_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z50_B5_tn.tif',
]

image_paths3 = [    
    'E:/NCKH/data/LANDSAT/HN/z50_B2_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z50_B3_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z50_B4_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z50_B5_hn.tif',
]

all_image_paths = [image_paths1, image_paths2, image_paths3]

brands = []



def load_and_prepare(image_paths):
    bands = []
    for path in image_paths:
        with rasterio.open(path) as src:
            band = src.read()  # (1, H, W)
            print(band.shape)
            bands.append(band)
    data = np.concatenate(bands)            # (n_bands, H, W)
    data = np.transpose(data, (1, 2, 0))             # (H, W, n_bands)
    w, h, c = data.shape
    data_2d = data.reshape(w * h, c)                 # (H*W, n_bands)
    return data_2d, w, h ,c 

# Đọc và xử lý từng nhóm ảnh riêng biệt
data1, w1, h1, c1 = load_and_prepare(image_paths1)
data2, w2, h2, c2 = load_and_prepare(image_paths2)
data3, w3, h3, c3 = load_and_prepare(image_paths3)

all_data = [data1, data2, data3]


# Bước 6: Phân cụm bằng Fuzzy C-Means
C = 6
M = 2
EPSILON = 1e-5
MAXITER = 10000
dcfcm = Dcfcm(n_clusters=C, m=M, epsilon=EPSILON, max_iter=MAXITER)
dcfcm.phase1(all_data)
dcfcm.phase2(iterations=10000)

# dcfm = Dcfcm(n_clusters=6, m=2, epsilon=1e-5, max_iter=10000)  # Khởi tạo Dcfcm
# centroids, membership_matrix, steps, _ = dcfm.fit(data)  # Sử dụng Dcfcm để phân cụm


# Bảng màu cho từng lớp
COLORS = np.array([
    [0, 0, 255],       # Class 1: Sông, hồ
    [128, 128, 128],   # Class 2: Đất trống, đường
    [0, 255, 0],       # Class 3: Cánh đồng, cỏ
    [1, 192, 255],     # Class 4: Rừng thưa, cây thấp
    [0, 128, 0],       # Class 5: Cây lâu năm
    [0, 64, 0]         # Class 6: Rừng rậm
], dtype=np.uint8)


# Bước 7: Tạo ảnh phân đoạn
# labels = np.argmax(membership_matrix, axis=1)  # lấy nhãn theo giá trị thành viên cao nhất
# labels_image = labels.reshape((height, width))
# # Bước 8: Gán màu theo từng nhãn
# colored_segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
# for i, color in enumerate(COLORS):
#     colored_segmented_image[labels_image == i] = color
def create_segmented_image(membership_matrix, width, height, colors):
    labels = np.argmax(membership_matrix, axis=1)
    labels_image = labels.reshape((width,height))  # Chú ý chiều là (H, W)
    segmented_image = np.zeros((width,height, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        segmented_image[labels_image == i] = color
    return segmented_image


sizes = [(w1, h1), (w2, h2), (w3, h3)]
titles = ['SCL (Site 1)', 'TN (Site 2)', 'HN (Site 3)']


# Bước 9: Hiển thị ảnh phân đoạn
# plt.figure(figsize=(8, 8))
# plt.imshow(colored_segmented_image)
# plt.title("Ảnh phân đoạn theo Fuzzy C-Means")
# plt.axis('off')
# plt.show()

plt.figure(figsize=(15, 5))

# Duyệt qua từng site và hiển thị ảnh phân đoạn
for i, site in enumerate(dcfcm.data_site):
    width, height = sizes[i]
    membership_matrix = site.membership_matrix
    segmented_image = create_segmented_image(membership_matrix, width, height, COLORS)
    
    plt.subplot(1, 3, i + 1)
    plt.imshow(segmented_image)
    plt.title(f"Ảnh phân đoạn - {titles[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()