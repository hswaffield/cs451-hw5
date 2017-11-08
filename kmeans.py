# Henry Swaffield & Hans Goudey - CS451 - Fall 2017

import numpy as np

# lists, with an entry for each training example:
# X[m] - each a 16d input vector
# C[m] - a number, j, which cluster the example is closest to

# mu[k] - 16d vector, which countries correspond to the cluster, K

def read_csv():
    
    data = []
    with open('country.csv') as csvfile:
        countryReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in countryReader:
            data.append(np.array(row))
        print(data[1])
        print(data.shape)

def squared_distance(v1, v2):
    output = 0
    for i in range(len(v1)):
        output += v1[i] * v2[i]
    return output
    
def find_centroid(x, mu):
    distances = [squared_distance(x[i], mu[i]) for i in range(len(x))]
    smallest_distance = distances.index(min(distances))
    
    return smallest_distance

def main():
    read_csv()


if __name__ == '__main__':
    main()
