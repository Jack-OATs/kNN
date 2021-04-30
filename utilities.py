import numpy as np
from random import shuffle

def str_to_float(data, index):
    for row in data:
        row[index] = float(row[index].strip())


def str_to_int(data, index):
    vals = [row[index] for row in data]
    unique = set(vals)
    cl = {}
    for i, v in enumerate(unique):
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
        str_to_float(data, i)
    lk = str_to_int(data, len(data[0])-1)
    # print("data now looks like {} and dictionary is {}".format(data[0], lk))
    return data


def parse_csv(file_path):
    data = load_csv(file_path)
    features = [row[:-1] for row in data]
    classes = [row[-1] for row in data]
    # print('features are {} and classes are {}'.format(features, classes))
    return features, classes


def k_folds(dataset, k):
    full_data = list(zip(dataset[0], dataset[1]))
    shuffle(full_data)
    folds = np.array_split(full_data, k)
    final_fold = []

    for i in range(len(folds)):
        training_features = []
        training_classes = []
        for j in range(len(folds)):
            if i != j:
                for item in folds[j]:
                    training_features.append(item[0])
                    training_classes.append(item[1])
        testing_examples = []
        testing_classes = []
        for item in folds[i]:
            testing_examples.append(item[0])
            testing_classes.append(item[1])
        final_fold.append(((training_features, training_classes), (testing_examples, testing_classes)))
    return final_fold



if __name__ == '__main__':
    parse_csv("Iris_Flower_Species.csv")