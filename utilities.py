import numpy as np


def parse_csv(file_path):
    data_set = open(file_path, 'r')
    data = data_set.read()
    data_set.close()
    rows = data.split('\n')
    data_arr = np.array([[c for c in r.split(',')]for r in rows])
    classes = data_arr[:,-1]
    features = data_arr[:,:-1]
    return features, classes


if __name__ == '__main__':
    parse_csv("Iris_Flower_Species.csv")