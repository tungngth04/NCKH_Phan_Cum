# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import time
# start_time = time.time()
# #Tạo dữ liệu gồm 300 điểm dữ liệu, với 3 cụm
# data, labels = make_blobs(n_samples=300, centers=3, random_state=0)
# #Khởi tạo mô hình với số cụm là 3 và số lần lặp là 300
# kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=0)
# # Huấn luyện mô hình với dữ liệu đầu vào (tìm ra 3 tâm cụm tối ưu nhất)
# kmeans.fit(data)
# #Gán nhãn
# predictions = kmeans.predict(data)
# end_time = time.time()
# plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap="viridis")
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="x", label="Cluster")
# plt.title("K-means Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.show()

# print("Time: " + str(end_time - start_time))
# print("Labels:", labels)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import time

# Đọc dữ liệu từ tệp CSV (không có nhãn cột)
file_path = 'E:\\NCKK\\K_Means\\109_wine.data.csv'
data = pd.read_csv(file_path, header=None)

# Tạo các nhãn ngẫu nhiên cho dữ liệu với số lượng điểm bằng số dòng trong file CSV
n_samples = data.shape[0]  # Số điểm dữ liệu
n_features = data.shape[1] - 1  # Số đặc trưng (trừ cột Class)

# Tạo nhãn ngẫu nhiên bằng make_blobs
_, labels = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=0)

# Khởi tạo và huấn luyện mô hình KMeans
start_time = time.time()
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=0)
kmeans.fit(data.iloc[:, 1:])  # Dữ liệu không bao gồm cột đầu tiên (nếu là nhãn Class)
predictions = kmeans.predict(data.iloc[:, 1:])
end_time = time.time()

# Vẽ biểu đồ phân cụm (dùng hai đặc trưng đầu tiên để trực quan hóa)
plt.scatter(data.iloc[:, 1], data.iloc[:, 2], c=predictions, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', label='Cluster centers')
plt.title('K-means Clustering with Random Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# In ra thời gian thực hiện và nhãn ngẫu nhiên
print("Thời gian thực hiện: {:.4f} giây".format(end_time - start_time))
print("Nhãn ngẫu nhiên:\n", labels)
