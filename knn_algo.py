import math


def euclidean(all_x, all_y):
    all_points = zip(all_x, all_y)
    set_diff = []
    for x, y in all_points:
        set_diff.append(math.pow(x-y, 2))
    return math.sqrt(math.fsum(set_diff))
