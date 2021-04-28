import numpy as np
from csv import reader


def str_to_float(data, index):
    for row in data:
        row[index] = float(row[index].strip())


def str_to_int(data, index):
    vals = [row[index] for row in data]
    cl = {}
    for i, v in enumerate(vals):
        cl[v] = i
    for row in data:
        row[index] = cl[row[index]]
    return cl


def load_csv(file_path):
    og_file = open(file_path, 'r')
    data_set = og_file.read()
    og_file.close()
    rows = data_set.split('\n')
    data = []
    for row in rows:
        row = row.split(',')
        sub_data = []
        for sd in row:
            sub_data.append(sd)
        data.append(sub_data)
    for i in range(len(data[0])-1):
        str_to_float(data, len(data)-1)
    lk = str_to_int(data, len(data)-1)
    print("data now looks like {} and dictionary is {}".format(data[0], lk))
    return data


def parse_csv(file_path):
    data = load_csv(file_path)
    classes = [row[:-1] for row in data]
    features = [row[-1] for row in data]
    # print('features are {} and classes are {}'.format(features, classes))
    return features, classes


if __name__ == '__main__':
    parse_csv("Iris_Flower_Species.csv")