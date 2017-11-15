# Henry Swaffield & Hans Goudey - CS451 - Fall 2017

import csv
import numpy as np
import random
import matplotlib.pyplot as plt

# X[m] - each a 16d input vector data point for a country
# C[m] - index of the closest centroid
# mu[k] - 16d vector, which countries correspond to the location of a centroid


def read_csv(filename):
    # Reads data points into a list of numpy arrays
    x = []
    with open(filename) as csvfile:
        country_reader = csv.reader(csvfile, delimiter=',')
        next(country_reader, None)  # Skipping the header
        for row in country_reader:
            row = [float(row[i]) for i in range(1, len(row))]
            x.append(np.array(row))
    return x

def squared_distance(v1, v2):
    return np.sum((v2 - v1)**2)
    
def find_nearest_centroid(xi, mu):
    # Returns the index of the closest centroid to the input data vector
    distances = [squared_distance(xi, mu[i]) for i in range(len(mu))]
    return distances.index(min(distances))

def get_centroids(x, mu):
    # Gets the index of the closest centroid for every data point in x
    return [find_nearest_centroid(x[i], mu) for i in range(len(x))]

def cost(x, c, mu):
    # Returns the kmeans algorithm cost for the current centroid assignments and locations
    return np.mean([squared_distance(x[i], mu[c[i]]) for i in range(len(x))])

def kmeans(x, k):
    mu = random.sample(x, k=k)
    c = []
    # Run the kmeans algorithm, breaking out of the loop if the cost converges
    last_cost = cost(x, get_centroids(x, mu), mu)
    for iterations in range(0, 500):
        # Cluster Assignment
        c = get_centroids(x, mu)

        # Update Centroids
        for i in range(len(mu)):
            mu_sum = np.zeros(16)
            count = 0
            # Loop through the data to find the total of the points assigned to the current centroid
            for index in range(len(c)):
                if c[index] == i:
                    mu_sum = mu_sum + x[index]
                    count += 1
            # The centroid becomes the mean of all of its assigned data
            mu[i] = (mu_sum / count)

        this_cost = cost(x, c, mu)
        if this_cost == last_cost:
            # The algorithm has converged
            break
        last_cost = this_cost

    return last_cost, c, mu

def run_random_kmeans(x, k):
    print('Running randomized k_means, k: ', k)
    count = 0
    best_c = []
    mu = []
    lowest_cost = kmeans(x, k)[0]
    this_cost = 0
    # Run kmeans 100 times, keeping track of the best cost and how many times it is generated
    for i in range(100):
        this_cost, c, mu = kmeans(x, k)
        if this_cost == lowest_cost:
            count += 1
        elif this_cost < lowest_cost:
            lowest_cost = this_cost
            count = 1
            best_c = c

    return best_c, count, this_cost, mu


def main():
    x = read_csv('country.csv')

    # for k in range(1,6):
    #     best_c, count, cost, mu = run_random_kmeans(x, k)
    #     print('k:', k, '('+str(count)+'/100)', 'cost:', cost, 'Size of USA's cluster:', best_c.count(best_c[185]))

    # The number of k values to go through
    max_k = 30

    # Store the tuple list of results from the algorithm
    results = [run_random_kmeans(x, k) for k in range(1, max_k + 1)]

    # Plot results:
    k_values = [i + 1 for i in range(max_k)]
    costs = [results[i][2] for i in range(max_k)]
    mu = [results[i][3] for i in range(max_k)]
    centroids = [results[i][0] for i in range(max_k)]
    plt.plot(k_values, costs, 'ro')

    # Write the data to a csv file for analysis externally
    with open('results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['k_values', 'costs', 'centroid', 'mu'])
        writer.writeheader()
        for i in range(max_k):
            print(i)
            current_row = {'k_values': k_values[i], 'costs': costs[i], 'centroid': centroids[i], 'mu': mu[i]}
            writer.writerow(current_row)
    plt.show()


if __name__ == '__main__':
    main()
