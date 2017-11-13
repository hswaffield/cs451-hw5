# Henry Swaffield & Hans Goudey - CS451 - Fall 2017

import csv
import numpy as np

# lists, with an entry for each training example:
# X[m] - each a 16d input vector
# C[m] - a number, j, which cluster the example is closest to

# mu[k] - 16d vector, which countries correspond to the cluster, K

def read_csv(filename):
    x = []
    with open(filename) as csvfile:
        country_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(country_reader, None)
        for row in country_reader:
            row = [float(row[i]) for i in range(1, len(row))]
            x.append(np.array(row))
    return x

def squared_distance(v1, v2):
    assert (len(v1) == len(v2)), "Vectors are not the same length.\nv1: %r\nv2: %r" % (v1, v2)
    output = 0
    for i in range(len(v1)):
        output += v1[i] * v2[i]
    return output
    
def find_nearest_centroid(xi, mu):
    distances = [squared_distance(xi, mu[i]) for i in range(len(mu))]
    return distances.index(min(distances))

def get_centroids(x, mu):
    return [find_nearest_centroid(x[i], mu) for i in range(len(x))]

def cost(x, c, mu):
    print('c:', c)
    print('mu:', mu)
    return np.mean([squared_distance(x[i], mu[c[i]]) for i in range(len(x))])


def kmeans(x, k):
    print('X:', x)
    mu = x[:k]
    print('MU: ', mu)
    lastCost = 99999999999999999999  #cost(x, get_centroids(x, mu), mu)
    for i in range(0, 500):
        # Cluster Assignment
        c = get_centroids(x, mu)

        # Update Centroids
        for i in range(len(mu)):
            # New mu[i] is the average of all of the data points assigned to it
            data_assigned_to_centroid = []
            for index in c:
                if index == i:
                    data_assigned_to_centroid.append(x[index])



            # ERROR AREA
            data_assigned_to_centroid = zip(data_assigned_to_centroid)
            for d in data_assigned_to_centroid:
                print(d)
            mu = np.array([np.mean(x) for x in zip(data_assigned_to_centroid)])  # THIS LINE IS PROBABLY CAUSING THE PROBLEM

        # Print Cost
        thisCost = cost(x, c, mu)
        if thisCost == lastCost:
            break




def main():
    x = read_csv('country.csv')
    kmeans(x, 1)


if __name__ == '__main__':
    main()
