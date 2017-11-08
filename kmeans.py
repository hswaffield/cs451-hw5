# Henry Swaffield & Hans Goudey - CS451 - Fall 2017

import csv
import numpy as np

# lists, with an entry for each training example:
# X[m] - each a 16d input vector
# C[m] - a number, j, which cluster the example is closest to

# mu[k] - 16d vector, which countries correspond to the cluster, K

def read_csv():
    
    x = []
    with open('country.csv') as csvfile:
        countryReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in countryReader:
            x.append(np.array(row))
        print(x[1])
        print(x)

def squared_distance(v1, v2):
    output = 0
    for i in range(len(v1)):
        output += v1[i] * v2[i]
    return output

# def get_centroid(country, centroids):
#     # initializing with a value to the nearest centroid.
#     min_distance = squared_distance(country, centroids[0])
#     nearest = 0
#     for centroid in range(len(centroids)):
#         distance = squared_distance(country, centroids[centroid])
#         if min_distance > distance:
#             min_distance = distance
#             nearest = centroid
#     return nearest
    
def find_nearest_centroid(xi, mu):
    distances = [squared_distance(xi[i], mu[i]) for i in range(len(xi))]
    smallest_distance = distances.index(min(distances))
    
    return smallest_distance


def get_centroids(x, mu):
    return [find_nearest_centroid(x[i], mu) for i in range(len(x))]

def compute_cost(x, c, mu):
    return np.mean([squared_distance(x[i], mu[c[i]]) for i in range(len(x))])

def main():
    read_csv()


if __name__ == '__main__':
    main()
