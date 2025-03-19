import time
import numpy as np
from FCM import FuzzyCMeans
import pandas as pd
from Ultility.ultility import divide_data_for_collaborativ
class Dcfcm():
    def __init__(self, n_clusters: int, m: float = 2.0, beta: float = 0.5, epsilon: float = 1e-5, max_iter: int = 10000):
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.data_site = [] #Lưu trữ các datasite là các đối tượng trong FCM
        self.beta = beta #hệ só cộng tác giữa các datasite 

    def phase1(self, datas: list):
        #Chạy FCM cho các datasite
        self.len_datasite = len(datas)
        for data in datas:
            fcm = FuzzyCMeans(self.n_clusters, self.m, self.epsilon, self.max_iter)
            fcm.fit(data)
            self.data_site.append(fcm)
  
    def caculate_U_fall(self):
        N, _ = self.data_site[0].local_data.shape
        c = self.data_site[0].cluster_centers.shape[0]
        # print("N,c", N,c)
        u_fall = np.zeros((self.len_datasite, self.len_datasite, N, c ))
        for i in range(len(self.data_site)):
            # print(self.data_site[i].local_data)
            for j in range(len(self.data_site)):
                if i == j:
                    continue
                # print(f"Shape of data_site[i].local_data: {self.data_site[i].local_data.shape}")
                # print(f"Shape of data_site[j].cluster_centers: {self.data_site[j].cluster_centers.shape}")

                u_fall[i][j] = FuzzyCMeans._update_membership_matrix(self,data=self.data_site[i].local_data, centroids=self.data_site[j].cluster_centers)
                
        return u_fall

    def caculate_U(self, u_fall , fcm_i):
        component1 = fcm_i._update_membership_matrix(fcm_i.local_data, fcm_i.cluster_centers)
        
        component3_1 = self.beta * np.sum(u_fall, axis=0) 
        component3_2 = 1 + self.beta * (self.len_datasite - 1)

        component3 = component3_1 / component3_2

        component2 = 1 - np.sum(component3, axis=1, keepdims=True)

        return (component1 * component2) +  component3

    def caculate_v(self,  U_ii: np.array, u_fall, data: np.array):
        # print("U_ii shape:", U_ii.shape)
        # print("u_fall shape:", u_fall.shape)
        # print("signal shape before fix:", ((U_ii[np.newaxis, :] - u_fall) ** 2).shape)

        signal = (U_ii[np.newaxis,:] - u_fall) ** 2

        denominator1 = np.sum(U_ii ** 2, axis=0, keepdims=True)
        denominator2 = self.beta * np.sum(np.sum(signal, axis=1), axis=0, keepdims=True)
        
        # numerator1 = np.sum((U_ii ** 2) * data, axis=0, keepdims=True)
        numerator1 = np.sum((U_ii ** 2)[:, :, np.newaxis] * data[:, np.newaxis, :], axis=0)

        # numerator2 = self.beta * np.sum(np.sum(signal * data, axis=1), axis=0, keepdims=True)
        numerator2 = self.beta * np.sum(np.sum(signal[:,:,:, np.newaxis] * data[np.newaxis, :, np.newaxis,:], axis=1), axis=0, keepdims=True)

        # return (numerator1 + numerator2) / (denominator1 + denominator2)
        return (numerator1 + numerator2) / (denominator1 + denominator2)[:,:, np.newaxis]


    def phase2(self, iterations: int):
        check = [False] * self.len_datasite
        for _ in range(iterations):
            u_fall = self.caculate_U_fall()  # Tính ma trận cảm ứng giữa các site
            for i in range(self.len_datasite):
                fcm_i = self.data_site[i]
                # Cập nhật ma trận thành viên dựa trên U_fall
                new_membership_matrix = self.caculate_U(u_fall[i], fcm_i)
        
                # Cập nhật lại trọng tâm cụm
                # fcm_i.cluster_centers = self.caculate_v(new_membership_matrix, u_fall[i], fcm_i.local_data)
                new_centers = self.caculate_v(new_membership_matrix, u_fall[i], fcm_i.local_data)
                fcm_i.cluster_centers = np.squeeze(new_centers)  # Loại bỏ chiều thừa


                if (np.linalg.norm(new_membership_matrix - fcm_i.membership_matrix) < self.epsilon):
                    check[i] = True
                    continue
                fcm_i.membership_matrix = new_membership_matrix
            if all(check):
                break


import pandas as pd
from Ultility.validity import dunn, davies_bouldin, calinski_harabasz, silhouette, separation, classification_entropy, hypervolume, cs, partition_coefficient, partition_entropy, f1_score, accuracy_score
from Ultility.data import  round_float , fetch_data_from_uci

if __name__ == "__main__":
    data_dict = pd.read_csv(r"/NCKH/data/UCI/Iris.csv")
    data = data_dict.iloc[:, 1:-1].values
    num_sites = 3
    sub_datasets = divide_data_for_collaborativ(data, num_sites)

    # Khởi tạo CFM với 3 cụm (Iris có 3 lớp)
    dcfcm = Dcfcm(n_clusters=3, beta=0.5, max_iter=100)

    # Chạy giai đoạn 1 (Phân cụm cục bộ tại mỗi site)   
    dcfcm.phase1(sub_datasets)

    # In ra trọng tâm cụm của từng data site
    print("\n Trọng tâm cụm trước khi cộng tác:")
    for i, site in enumerate(dcfcm.data_site):
        print(f"Data site {i+1}:\n", site.cluster_centers)

    # Giai đoạn 2: Cập nhật theo U_fall
    start_time = time.time()
    dcfcm.phase2(iterations=5)
    phase2_time = time.time() - start_time

    print("\n Trọng tâm cụm sau khi cộng tác:")
    for i, site in enumerate(dcfcm.data_site):
        print(f"Data site {i+1}:\n", site.cluster_centers)


    # u_fall = dcfcm.caculate_U_fall()
    # print("Ma trận u_fall:")
    # print(u_fall)

    # đầu ra là cặp u v cho từng cặp dataasite và chỉ số của từng bộ datasite


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
    
    titles = ['Alg', 'Time', 'Step', 'DB-', 'PC+', 'CE-', 'S- ' , 'CH+', 'SI+', 'FHV+', 'CS-', 'F1+', 'AC+']
    print(SPLIT.join(titles))
    for i, site in enumerate(dcfcm.data_site):
        print(print_info(title=f"Site-{i+1}", X=site.local_data, U=site.membership_matrix, V=site.cluster_centers, process_time=phase2_time, step=5))

