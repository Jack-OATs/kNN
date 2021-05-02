import utilities
import knn_algo
from knn_algo import OutcomeMetrics
import numpy as np


def run_knn():
    data = utilities.parse_csv('Iris_Flower_Species.csv')
    folds = utilities.k_folds(data, 5)
    accuracies = []
    predictions_list = []
    for fold in folds:
        train_data, test_data = fold
        train_features, train_classes = train_data
        # fc_train = zip(train_features, train_classes)
        test_features, test_classes = test_data
        for test in test_features:
            predictions = knn_algo.classify(train_features, train_classes, test)
            predictions_list.append(predictions)
        ocm = OutcomeMetrics(predictions_list, train_classes)
        # print("predictions list is {}\nactual list is {}\n".format(predictions_list, train_classes))
        accuracies.append(ocm.accuracy())
    print("average of accuracy on classification = " + str(np.average(accuracies)))

if __name__=='__main__':
    run_knn()