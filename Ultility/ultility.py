import numpy as np
def divide_data_for_collaborativ(data, n_data_site):
    #Xáo trộn bọ dữ liệu rồi đem chia đều cho n_data_site
    np.random.seed(42)
    np.random.shuffle(data)
    sub_data = np.array_split(data, n_data_site, axis=0)
    
    return sub_data