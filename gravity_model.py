import numpy as np
import matplotlib.pyplot as plt

dims = 500
max_popn = 10000
nlocs = 7


def find_locations_and_populations(dims, max_pop, nlocs):

    locations = np.zeros((nlocs, 2))
    populations = np.zeros((nlocs, 1))
    for i in range(nlocs):
        coords = [np.random.randint(50, 450),  np.random.randint(50, 450)]
        population = np.random.randint(0, max_pop)
        locations[i] = coords
        populations[i] = population

    return locations, populations


def calculate_distance_matrix(nlocs, locations):

    distances = np.zeros((nlocs, nlocs))
    x = list(locations[:, 0])
    y = list(locations[:, 1])
    for k in range(nlocs):
        for l in range(nlocs):
            deltax = x[k] - x[l]
            deltay = y[k] - y[l]
            dist = np.sqrt(deltax**2 + deltay**2)
            distances[k][l] = dist

    return distances


def plot_locations(locations, pops, dims, OD):

    x = list(locations[:, 0])
    y = list(locations[:, 1])
    plt.scatter(x, y, 0.1*pops)
    labels = list(range(dims))
    for label, i, j, in zip(labels, x, y):
        plt.annotate(label+1, xy = (i,j))
    plt.ylim([0, dims])
    plt.xlim([0, dims])
    plt.show()


def find_likelihoods(populations, distances):

    dim = len(distances)
    probs = np.zeros((dim, dim))
    likelihoods = np.zeros((dim, dim))

    for i in range(dim):
        iprobs = np.zeros((dim))
        for j in range(dim):
            if i != j:
                mass_j = int(populations[j])
                dist_ij = distances[i, j]
                probs[i, j] = mass_j / dist_ij**2
                iprobs[j] = probs[i, j]
        likelihoods[i] = iprobs/sum(iprobs)

    return likelihoods


def create_OD_matrix(populations, likelihoods, dims):

    OD_matrix = np.zeros((dims, dims))

    for i in range(dims):
        OD_matrix[i] = np.rint(0.1 * populations[i] * likelihoods[i])

    return OD_matrix



locations, populations = find_locations_and_populations(dims, max_popn, nlocs)

distances = calculate_distance_matrix(nlocs, locations)

likelihoods = find_likelihoods(populations, distances)

OD = create_OD_matrix(populations, likelihoods, nlocs)

print(populations)
print(OD)

plot_locations(locations, populations, dims, OD)







