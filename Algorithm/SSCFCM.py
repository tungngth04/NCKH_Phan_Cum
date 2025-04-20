import time
import numpy as np
from CFCM import Dcfcm

class Dssfcm(Dcfcm):
    def __init__(self, n_clusters: int, m: float = 2.0, beta: float = 0.5, epsilon: float = 1e-5, max_iter: int = 10000, index: int = 0):
        super().__init__(n_clusters, m, beta, epsilon, max_iter)
        self.beta = 0.5

    def update_centroids():
        