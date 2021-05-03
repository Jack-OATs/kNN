import math


class OutcomeMetrics:
    """
    This class determines the recall, precision, as well as for accuracy for
    the predicted labels in comparison with the actual labels.
    """

    def __init__(self, classifier_labels, actual_labels):
        """
        Initialize the class when the object is called.
        :param classifier_labels: The predicted labels of the algorithm.
        :param actual_labels: The expected labels that were pre determined.
        """
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
    """
    Determines the distance between the two single points of two separate
    lists that are in a euclidean space.
    :param all_x: A list of values for one point
    :param all_y: A list of values for another point
    :return: The square root of the sum of all the points.
    """
    all_points = zip(all_x, all_y)
    set_diff = []
    for x, y in all_points:
        set_diff.append(math.pow(x-y, 2))
    return math.sqrt(math.fsum(set_diff))


def nearest_neighbor(train_features, train_classes, test_features, depth):
    """
    Determines the nearest neighbors based off of two points euclidean distance.
    :param train_features: All of the feature lists used to train the algorithm.
    :param train_classes: All of the classifications that are in parallel with every list of features.
    :param test_features: A single list of features used to test the algorithm.
    :param depth: How many neighbors should be returned.
    :return: A list of neighbors.
    """
    dist_list = []
    fc_test = zip(train_classes, train_features)
    for cl, row in fc_test:
        dist = euclidean(row, test_features)
        dist_list.append((dist, test_features, cl))
    dist_list.sort(key=lambda d: d[0])
    neighbor = []
    for i in range(depth):
        neighbor.append(dist_list[i][2])#, dist_list[i][2]))
    return neighbor


def classify(train_features, train_classes, test_features, depth=30):
    """
    Predicts what should be the best classification for a test based on the mode of classes received from
    the neighbors.
    :param train_features: All of the feature lists used to train the algorithm.
    :param train_classes: All of the classifications that are in parallel with every list of features.
    :param test_features: A single list of features used to test the algorithm.
    :param depth: How many neighbors should be returned.
    :return: The predicted value.
    """
    neighbors = nearest_neighbor(train_features, train_classes, test_features, depth)
    class_vals = [cl for cl in neighbors]
    pred = max(class_vals, key=class_vals.count)
    return pred
