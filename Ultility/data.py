import os
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl

TEST_CASES = {
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 4,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


def round_float(number: float) -> float:
    return round(number, 3)


def euclidean_distances(A: np.ndarray, B: np.ndarray, axis: int = None):
    return np.linalg.norm(A - B, axis=axis)

def load_dataset(data: dict, file_csv: str = ''):
    # label_name = data['data']['target_col']
    print('uci_id=', data['data']['uci_id'])  # Mã bộ dữ liệu
    print('data name=', data['data']['name'])  # Tên bộ dữ liệu
    print('data abstract=', data['data']['abstract'])  # Tên bộ dữ liệu
    print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=0)
    print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values  # Cột cuối là nhãn
    return features, labels


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        response = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        features, labels = load_dataset(data=json.load(response))
        return {'X': features, 'y': labels}  # Trả về cả features và labels
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')