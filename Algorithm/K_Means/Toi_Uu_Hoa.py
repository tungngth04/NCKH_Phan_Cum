import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import time
#Với thuật toán K Means++
start_time = time.time()
 # Tạo dữ liệu gồm 300 điểm dữ liệu, với 3 cụm
data, labels = make_blobs(n_samples=300, centers=3, random_state=0)
# Khởi tạo mô hình với số cụm là 3 và số lần lặp là 300, sử dụng K-means++
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
# Huấn luyện mô hình với dữ liệu đầu vào (tìm ra 3 tâm cụm tối ưu nhất)
kmeans.fit(data)
# Gán nhãn
predictions = kmeans.predict(data)
end_time = time.time()
plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="x", label="Cluster")
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("Time: " + str(end_time - start_time))
print("Labels:", labels)

# Với thuật toán Mini-Batch K-means
start_time = time.time()
# Khởi tạo mô hình với số cụm là 3 và số lần lặp là 300, sử dụng Mini-Batch K-means
mini_batch_kmeans = MiniBatchKMeans(n_clusters=3, n_init=10, max_iter=300, random_state=0)
# Huấn luyện mô hình với dữ liệu đầu vào (tìm ra 3 tâm cụm tối ưu nhất)
mini_batch_kmeans.fit(data)
#Gán nhãn
mini_batch_predictions = mini_batch_kmeans.predict(data)

end_time = time.time()

plt.scatter(data[:, 0], data[:, 1], c=mini_batch_predictions, cmap="viridis")
plt.scatter(mini_batch_kmeans.cluster_centers_[:, 0], mini_batch_kmeans.cluster_centers_[:, 1], c="red", marker="x", label="Cluster")
plt.title("Mini-Batch K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("Mini-Batch Time: " + str(end_time - start_time))
print("Mini-Batch Labels:", labels)
