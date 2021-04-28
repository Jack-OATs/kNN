import math
import numpy as np

def euclidean(all_x, all_y):
    all_points = zip(all_x, all_y)
    set_diff = []
    for x, y in all_points:
        set_diff.append(math.pow(x-y, 2))
    return math.sqrt(math.fsum(set_diff))


def nearest_neighbor(train_data, test_data, num_of_neighbor):
        dist_list = []
        for row in train_data:
            dist = euclidean(row, test_data)
            dist_list.append((dist, test_data))
        dist_list.sort(key=lambda d: d[0])
        neighbor = []
        for i in range(num_of_neighbor):
            neighbor.append(dist_list[i][1])
        return neighbor


def classify(train_data, test_data, num_of_neighbor):
    neighbors = nearest_neighbor(train_data, test_data, num_of_neighbor)
    class_vals = [neighbor[-1] for neighbor in neighbors]
    pred = [lambda vals: max(vals, key=vals.count),
            lambda vals: np.average(vals)]
    preds = []
    for p in pred:
        preds.append(p(class_vals))
    return preds


def get_example():
    return [[2.7810836,2.550537003,0],
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]]


if __name__ == '__main__':
    # n = nearest_neighbor(get_example(), get_example()[0], 3)
    # for i in n:
    #     print(i)
    for example in get_example():
        pred = classify(get_example(), example, 3)
        print("pred is {}".format(pred))