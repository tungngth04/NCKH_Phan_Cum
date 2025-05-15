import numpy as np
import matplotlib.pyplot as plt
import rasterio
from Algorithm.FCM import FuzzyCMeans
import os
from Algorithm.CFCM import Dcfcm 
from Ultility.test import align_clusters
from Algorithm.SSFCM import Dssfcm

# Bước 1: Đường dẫn đến các ảnh vệ tinh .tif
image_paths_scl = [
    'E:/NCKH/data/LANDSAT/SCL/z200_B2_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z200_B3_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z200_B4_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z200_B5_scl.tif',
]

image_paths_tn = [
    'E:/NCKH/data/LANDSAT/TN/z200_B2_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z200_B3_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z200_B4_tn.tif',
    'E:/NCKH/data/LANDSAT/TN/z200_B5_tn.tif',
]

image_paths_hn = [    
    'E:/NCKH/data/LANDSAT/HN/z200_B2_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z200_B3_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z200_B4_hn.tif',
    'E:/NCKH/data/LANDSAT/HN/z200_B5_hn.tif',
]
image_paths4 = [
    'E:/NCKH/data/LANDSAT/SCL/z50_B2_scl.tif',
    # 'E:/NCKH/z50_B5_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B3_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B4_scl.tif',
    'E:/NCKH/data/LANDSAT/SCL/z50_B5_scl.tif',
]
all_image_paths = [image_paths_scl, image_paths_tn, image_paths_hn]

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
data1, w1, h1, c1 = load_and_prepare(image_paths_scl)
data2, w2, h2, c2 = load_and_prepare(image_paths_tn)
data3, w3, h3, c3 = load_and_prepare(image_paths_hn)
data4, w4, h4, c4 = load_and_prepare(image_paths4)
all_data = [data1, data2, data3]


# Bước 6: Phân cụm bằng Fuzzy C-Means
C = 6
M = 2
EPSILON = 1e-5
MAXITER = 10000

fcm = FuzzyCMeans(n_clusters=C, m=M, epsilon=EPSILON, max_iter=MAXITER)
centroid4,membership4 , step , _= fcm.fit(data=data4)
print(membership4.shape)
print(centroid4.shape)
# Bảng màu cho từng lớp
COLORS = np.array([
    [0, 0, 255],        # Class 1: Rivers, lakes, ponds
    [128, 128, 128],    # Class 2: Vacant land, roads
    [0, 255, 0],        # Class 3: Field, grass
    [1, 192, 255],      # Class 4: Sparse forest, low trees
    [0, 128, 0],        # Class 5: Perennial Plants
    [0, 64, 0],         # Class 6: Dense forest, jungle
], dtype=np.uint8)



labels4 = np.argmax(membership4, axis=1) # Nhãn

# labels_image = labels.reshape(( height,width))
labels_image4 = labels4.reshape(( w4,h4))

# colored_segmented_image = np.zeros(( height,width, 3), dtype=np.uint8)
colored_segmented_image4 = np.zeros(( w4,h4, 3), dtype=np.uint8)

for i, color in enumerate(COLORS):
    colored_segmented_image4[labels_image4 == i] = color  # labels của các đoạn với màu tương ứng

plt.imshow(colored_segmented_image4)
plt.title("Z50")
plt.axis('off')
plt.show()

dcfcm = Dcfcm(n_clusters=C, m=M, epsilon=EPSILON, max_iter=MAXITER)
dcfcm.phase1(all_data)

dcfcm.phase2(iterations=10000)
from scipy.spatial.distance import cdist
mapping = np.argmin(cdist(dcfcm.data_site[0].cluster_centers, centroid4), axis=1)
def create_segmented_image(membership_matrix, width, height, colors):
    labels = np.argmax(membership_matrix, axis=1)
    labels_image = labels.reshape((width,height))  # Chú ý chiều là (H, W)
    segmented_image = np.zeros((width,height, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        segmented_image[labels_image == i] = color
    return segmented_image


