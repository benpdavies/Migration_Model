import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
from statsmodels.graphics.api import abline_plot
from scipy import stats
from statsmodels import graphics
import matplotlib.pyplot as plt

dims = 500
max_popn = 10000
nlocs = 3


def find_locations_and_populations(max_pop, nlocs):

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


def find_likelihoods(populations, distances, mass_power, deterrance_power):

    dim = len(distances)
    probs = np.zeros((dim, dim))
    likelihoods = np.zeros((dim, dim))

    for i in range(dim):
        iprobs = np.zeros((dim))
        for j in range(dim):
            if i != j:
                mass_j = int(populations[j])
                dist_ij = distances[i, j]
                # probs[i, j] = mass_j**mass_power/dist_ij**-deterrance_power
                probs[i, j] = mass_j**mass_power * np.exp(-dist_ij /deterrance_power)
                iprobs[j] = probs[i, j]
        likelihoods[i] = iprobs/sum(iprobs)

    return likelihoods


def create_OD_matrix(populations, likelihoods, nlocs):

    OD_matrix = np.zeros((nlocs, nlocs))
    total_outflow = np.zeros(nlocs)

    for i in range(nlocs):

        # OD_matrix[i] = np.rint(0.1 * populations[i] * likelihoods[i])
        total_outflow[i] = np.random.randint(0, populations[i])
        OD_matrix[i] = np.rint(np.random.multinomial(total_outflow[i], likelihoods[i]))

    return total_outflow, OD_matrix


def main(nlocs, max_popn, mass_co, deterrance_co):

    locations, populations = find_locations_and_populations(max_popn, nlocs)

    distances = calculate_distance_matrix(nlocs, locations)

    likelihoods = find_likelihoods(populations, distances, mass_co, deterrance_co)
    # print(likelihoods)

    total_outflow, OD = create_OD_matrix(populations, likelihoods, nlocs)

    return distances, populations, OD.astype(int)

# print(total_outflow)
# print(populations)
# print(OD)

# distances, populations, OD = main(nlocs, max_popn)
# print(OD)

# plot_locations(locations, populations, dims, OD)



