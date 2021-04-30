import math
import numpy as np

class OutcomeMetrics:
    def __init__(self, classifier_labels, actual_labels):
        self.classifier_labels = classifier_labels
        self.actual_labels = actual_labels
        self.confusion_matrix = self.__build_confusion_matrix()

    def __build_confusion_matrix(self):
        # format should be [[true_positive, false_negative], [false_positive, true_negative]]
        all_labels = zip(self.classifier_labels, self.actual_labels)
        con_matrix = [[0, 0], [0, 0]]
        for cl, al in all_labels:
            if cl == al == 1:
                con_matrix[0][0] += 1
            elif cl == al == 0:
                con_matrix[1][1] += 1
            elif cl == 1 != al:
                con_matrix[1][0] += 1
            elif cl == 0 != al:
                con_matrix[0][1] += 1
        return con_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def precision(self):
        # precision is measured as: true_positive/ (true_positive + false_positive)
        con_mat = self.get_confusion_matrix()
        tp = con_mat[0][0]
        fp = con_mat[1][0]
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    def recall(self):
        # recall is measured as: true_positive/ (true_positive + false_negative)
        con_mat = self.get_confusion_matrix()
        tp = con_mat[0][0]
        fn = con_mat[0][1]
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)

    def accuracy(self):
        # accuracy is measured as:  correct_classifications / total_number_examples
        con_mat = self.get_confusion_matrix()
        tp = con_mat[0][0]
        tn = con_mat[1][1]
        total = math.fsum(con_mat[0]) + math.fsum(con_mat[1])
        if total == 0:
            return 0
        return (tp + tn) / total


def euclidean(all_x, all_y):
    all_points = zip(all_x, all_y)
    set_diff = []
    for x, y in all_points:
        set_diff.append(math.pow(x-y, 2))
    return math.sqrt(math.fsum(set_diff))


def nearest_neighbor(train_features, test_classes, test_features, depth):
        dist_list = []
        fc_test = zip(test_classes, train_features)
        for cl, row in fc_test:
            dist = euclidean(row, test_features)
            # print("dist is {}".format(dist))
            dist_list.append((dist, test_features, cl))
        dist_list.sort(key=lambda d: d[0])
        neighbor = []
        for i in range(depth):
            neighbor.append((dist_list[i][1], dist_list[i][2]))
        return neighbor


def classify(train_features, test_classes, test_features, depth=5):
    neighbors = nearest_neighbor(train_features, test_classes, test_features, depth)
    class_vals = [cl for _, cl in neighbors]
    # print("neighbors are {}".format(neighbors))
    # print("classes are {}\n".format(class_vals))
    pred = max(class_vals, key=class_vals.count)
    # print("pred is {}".format(pred))
    return pred


# if __name__ == '__main__':
#     # n = nearest_neighbor(get_example(), get_example()[0], 3)
#     # for i in n:
#     #     print(i)
#     for example in get_example():
#         pred = classify(get_example(), example, 6)
#         print("pred is {}".format(pred))