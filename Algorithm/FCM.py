import numpy as np
import time
from scipy.spatial.distance import cdist

class FuzzyCMeans:
    def __init__(self, n_clusters: int, m: float = 2.0, epsilon: float = 1e-5, max_iter: int = 10000):
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.cluster_centers = None
        self.membership_matrix = None
        self.process_time = 0 # Tính thời gian xử lý
    def _initialize_membership_matrix(self, n_points: int):
        np.random.seed(42)
        return np.random.dirichlet(np.ones(self.n_clusters), size=n_points)

    def _compute_cluster_centers(self, data: np.array, membership: np.ndarray = None):
        if membership is None:
            membership = self.membership_matrix
        membership_matrix_power = membership ** self.m
        return np.dot(membership_matrix_power.T, data) / membership_matrix_power.sum(axis=0)[:, None]

    def _update_membership_matrix(self, data: np.array, centroids: np.ndarray):
        distances =  cdist(data, centroids)

        inv_dists = distances ** (-2 / (self.m - 1))
        return inv_dists / np.sum(inv_dists, axis=1, keepdims=True)

    def fit(self, data: np.array):
        n_points, _ = data.shape
        _start_tm = time.time()
        self.membership_matrix = self._initialize_membership_matrix(n_points)

        for iteration in range(self.max_iter):
            self.cluster_centers = self._compute_cluster_centers(data,  self.membership_matrix)

            new_membership_matrix = self._update_membership_matrix(data, centroids=self.cluster_centers)
            
            if np.linalg.norm(new_membership_matrix - self.membership_matrix) < self.epsilon:
                break

            self.membership_matrix = new_membership_matrix
        self.process_time = time.time() - _start_tm
        return self.cluster_centers, self.membership_matrix, iteration + 1

import time
import numpy as np
from Ultility.data import round_float, TEST_CASES, fetch_data_from_uci
from Algorithm.FCM import FuzzyCMeans
from Ultility.validity import dunn, davies_bouldin, calinski_harabasz, silhouette, separation, classification_entropy, hypervolume, cs, partition_coefficient, f1_score

if __name__ == "__main__":
    _start_time = time.time()

    # Lấy dữ liệu từ UCI
    dataset_id = 602
    data_dict = fetch_data_from_uci(dataset_id)
    data = data_dict['X']
    C = TEST_CASES[dataset_id]['n_cluster']

    # Khởi tạo FuzzyCMeans và tính toán
    fcm = FuzzyCMeans(n_clusters=C)
    centroids, membership_matrix, steps = fcm.fit(data)

    print("Thời gian tính toán tuần tự", round_float(time.time() - _start_time))

    print("Centroids:\n", centroids)
    print("Số bước lặp:", steps)

    SPLIT = '\t'
    M = 2
    average='weighted'
    def wdvl(val: float) -> str:
        return str(round_float(val))

    def print_info(title: str, X: np.ndarray, U: np.ndarray, V: np.ndarray, process_time: float, step: int = 0, split: str = SPLIT) -> str:
        kqdg = [
            title,
            str(wdvl(process_time)),
            str(step),
            # wdvl(dunn(X, np.argmax(U, axis=1))),  # DI
            # wdvl(partition_entropy(U)),  # PE
            wdvl(davies_bouldin(X, np.argmax(U, axis=1))),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(classification_entropy(U)),  # CE
            wdvl(separation(X, U, V, M)),  # S
            wdvl(calinski_harabasz(X, np.argmax(U, axis=1))),  # CH
            wdvl(silhouette(X, np.argmax(U, axis=1))),  # SI
            wdvl(hypervolume(U, M)),  # FHV
            wdvl(cs(X, U, V, M)),  # CS
            # wdvl(f1_score(int_labels, predicted_labels, average))
        ]
        result = split.join(kqdg)
        return result
    
    titles = ['Alg', 'Time', 'Step', 'DB-', 'PC+', 'CE-', 'S-       ' , 'CH+        ', 'SI+', 'FHV+', 'CS-', 'F1+', 'AC+']
    print(SPLIT.join(titles))
    print(print_info( title='SSFCM', X=data, U=membership_matrix, V=centroids, process_time=fcm.process_time, step=steps))
