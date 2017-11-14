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
        output += (v1[i] - v2[i])**2
    return output
    
def find_nearest_centroid(xi, mu):
    distances = [squared_distance(xi, mu[i]) for i in range(len(mu))]
    return distances.index(min(distances))

def get_centroids(x, mu):
    return [find_nearest_centroid(x[i], mu) for i in range(len(x))]

def cost(x, c, mu):
    return np.mean([squared_distance(x[i], mu[c[i]]) for i in range(len(x))])


def kmeans(x, k):
    mu = x[:k]
    # print('MU: ', mu)
    lastCost = cost(x, get_centroids(x, mu), mu)
    print(lastCost)
    for iterations in range(0, 500):
        # Cluster Assignment
        c = get_centroids(x, mu)

        # Update Centroids (loops through k times:)
        for i in range(len(mu)):

            mu_average = np.zeros((16))
            count = 0

            for index in range(len(c)):
                if c[index] == i:
                    # Match:
                    # data_assigned_to_centroid.append(x[index])
                    mu_average = mu_average + x[index]
                    count += 1


            # ERROR AREA
            # updating centroids, by averaging:

            # data_assigned_to_centroid = zip(data_assigned_to_centroid)

            # mu_average = np.zeros(shape)

            # for d in data_assigned_to_centroid:
            #     print(d)
            #
            # mu = np.array([np.mean(x) for x in zip(data_assigned_to_centroid)])  # THIS LINE IS PROBABLY CAUSING THE PROBLEM

            mu[i] = (mu_average / count)

            # print('mu_average - after division: ', mu[i])

            # print(mu[i])
            # print(mu)
            # print(mu[0].shape)

        # Print Cost


        thisCost = cost(x, c, mu)
        print(thisCost)
        if thisCost == lastCost:
            print('converged after num iterations: ', iterations)
            break
        lastCost = thisCost




def main():
    x = read_csv('country.csv')
    for k in range(1,6):
        print('k = ', k)
        kmeans(x, k)

if __name__ == '__main__':
    main()
