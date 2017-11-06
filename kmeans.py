# Henry Swaffield & Hans Goudey - CS451 - Fall 2017

import numpy as np

# lists, with an entry for each training example:
# X[m] - each a 16d input vector
# C[m] - a number, j, which cluster the example is closest to

# mu[k] - 16d vector, which countries correspond to the cluster, K

def read_csv():
    data = np.genfromtxt('country.csv', delimiter=",", skip_header=1, usecols=(range(1, 17)))
    print(data)
    print(data[1])
    print(data.shape)


def main():
    read_csv()


if __name__ == '__main__':
    main()