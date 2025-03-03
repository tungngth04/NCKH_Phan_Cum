# import numpy as np
# from numpy import ndarray
# import time
# class K_MEANS():
#     def __init__(self,epsilon:float=1e-5,maxiter:int=10000):
#         self._epsilon = epsilon
#         self._maxiter = maxiter
#     def update_labels(self,data:ndarray,centroids:ndarray)->ndarray:
#         dis=np.sum((data[:,np.newaxis,:]-centroids[np.newaxis,:,:])**2,axis=2)
#         return np.argmin(dis, axis=1)
#     def kmeans(self,data:ndarray,C:int,seed:int=0)->tuple:
#         if seed > 0:
#             np.random.seed(seed=seed)
#         # Khởi tạo tâm cụm ngẫu nhiên
#         centroids = data[np.random.choice(len(data), C, replace=False)]
#         for step in range(self._maxiter):
#             # Gán nhãn cho các điểm dữ liệu
#             labels = self.update_labels(data, centroids)
#             # Cập nhật tâm cụm
#             new_c=np.array([data[labels==k].mean(axis=0) for k in range(C)])
#             # Kiểm tra hội tụ
#             if np.sum((new_c - centroids) ** 2) < self._epsilon:
#                 break
#             centroids = new_c
#         return labels, centroids, step + 1
# start_time = time.time()   
# np.random.seed(0)
# data = np.random.randn(300, 2)

# C = 3

# kmeans_model = K_MEANS()
# labels, centroids, num_iterations = kmeans_model.kmeans(data, C)
# end_time = time.time()
# # In kết quả
# print("Time: " + str(end_time - start_time))
# print("Labels:", labels)
# print("Centroids:", centroids)
# print("Number of iterations:", num_iterations)

# # import numpy as np

# # data = np.array([[1, 2],
# #                  [3, 4],
# #                  [5, 6]])

# # centroids = np.array([[1, 1],
# #                       [2, 2],
# #                       [3, 3]])

# # data_expanded = data[:, np.newaxis, :]
# # centroids_expanded = centroids[np.newaxis, :, :]

# # print("data_expanded:\n", data_expanded)
# # print("centroids_expanded:\n", centroids_expanded)
# # # Tính toán phép trừ giữa mảng đã được mở rộng
# # difference = data_expanded - centroids_expanded
# # print("difference:\n", difference)
# # print("Sum:\n", np.sum(difference,axis=2))

# # print("Min:\n",np.argmin(np.sum(difference,axis=2),axis=2))
import numpy as np
import pandas as pd
import time
from numpy import ndarray

# Lớp K_MEANS
class K_MEANS():
    def __init__(self, epsilon: float = 1e-5, maxiter: int = 10000):
        self._epsilon = epsilon
        self._maxiter = maxiter

    def update_labels(self, data: ndarray, centroids: ndarray) -> ndarray:
        dis = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dis, axis=1)

    def kmeans(self, data: ndarray, C: int, seed: int = 0) -> tuple:
        if seed > 0:
            np.random.seed(seed=seed)
        # Khởi tạo tâm cụm ngẫu nhiên
        centroids = data[np.random.choice(len(data), C, replace=False)]
        for step in range(self._maxiter):
            # Gán nhãn cho các điểm dữ liệu
            labels = self.update_labels(data, centroids)
            # Cập nhật tâm cụm
            new_c = np.array([data[labels == k].mean(axis=0) for k in range(C)])
            # Kiểm tra hội tụ
            if np.sum((new_c - centroids) ** 2) < self._epsilon:
                break
            centroids = new_c
        return labels, centroids, step + 1

# Đọc dữ liệu từ tệp CSV
file_path = 'E:/NCKK/K_Means/109_wine.data.csv'
data = pd.read_csv(file_path, header=None)

# Tách dữ liệu, bỏ cột đầu tiên (nếu là nhãn 'Class')
data = data.iloc[:, 1:].values  # Chỉ lấy dữ liệu (loại bỏ cột Class nếu có)

# Thiết lập số cụm C
C = 3

# Khởi tạo và chạy mô hình K-Means
start_time = time.time()
kmeans_model = K_MEANS()
labels, centroids, num_iterations = kmeans_model.kmeans(data, C)
end_time = time.time()

# In kết quả
print("Thời gian thực hiện: {:.4f} giây".format(end_time - start_time))
print("Nhãn (Labels):", labels)
print("Tâm cụm (Centroids):", centroids)
print("Số lần lặp (Iterations):", num_iterations)
