import numpy as np
import random
import matplotlib.pyplot as plt
import gravity_model

mass_coeff = 2
deter_coeff = 5

distances, populations, OD = gravity_model.main(10, 1000, mass_coeff, deter_coeff)


def prob_ij(i, j, distances, populations, mass_co, deter_co):



    nlocs = len(populations)
    totals = np.zeros(nlocs)

    for n in range(nlocs):
        if i != n:
#            totals[n] = np.exp(mass_co*np.log(populations[n]) - distances[i, n]/deter_co)
            totals[n] = np.exp(mass_co*np.log(populations[n]) - distances[i, n] * deter_co)
    total = sum(totals)


#    value = np.exp(mass_co*np.log(populations[j]) - distances[i, j]*(1/deter_co))
    value = np.exp(mass_co*np.log(populations[j]) - distances[i, j]*(deter_co))
    if i == j:
        value = 0
    prob = value/total
    return prob


def gradient_massco(OD, distances, populations, mass_co, deter_co):


    mass_grad = 0
    nlocs = len(OD)
    for i in range(nlocs):
        first = 0
        second = 0
        for j in range(nlocs):
            first += OD[i, j]*np.log(populations[j])
        for l in range(nlocs):
            second += np.log(populations[l]) * prob_ij(i, l, distances, populations, mass_co, deter_co)
        second = sum(OD[i]) * second
        value = first - second
        mass_grad += value

    return mass_grad


def gradient_deterco(OD, distances, populations, mass_co, deter_co):


    deter_grad = 0
    nlocs = len(OD)
    for i in range(nlocs):
        first = 0
        second = 0
        for j in range(nlocs):
            first += (OD[i, j] * distances[i, j])
        for l in range(nlocs):
            second += (prob_ij(i, l, distances, populations, mass_co, deter_co) * distances[i, l])
        second = sum(OD[i]) * second
#        value = first + second
        value = - first + second
        deter_grad += value

    return deter_grad


def likelihood_grad(populations, distances, pop_co, deter_co):

    pop_grad = gradient_massco(OD, distances, populations, pop_co, deter_co)
    deter_grad = gradient_deterco(OD, distances, populations, pop_co, deter_co)
    params_grad = [pop_grad[0], deter_grad[0]]
    return params_grad


params = np.array((1., 1. / 5))
learning_rate = 5e-6
epochs = 5000
convergence1 = np.zeros(epochs)
convergence2 = np.zeros(epochs)
beta = mass_coeff * np.ones(epochs)
R = deter_coeff * np.ones(epochs)
velocity1 = np.zeros(epochs+1)
velocity2 = np.zeros(epochs+1)
gamma = 0.9

for i in range(epochs):

    convergence1[i] = params[0]
    convergence2[i] = params[1]

    params_grad = likelihood_grad(populations, distances, params[0], params[1])
    delta = np.array([learning_rate * params_grad[0], learning_rate * params_grad[1]])
    if i%50 == 0:
        print(params)
#        print([params_grad, delta])
    # velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
    # velocity1[i + 1] = gamma * velocity1[i] + learning_rate * params_grad[0]
    # velocity2[i + 1] = gamma * velocity2[i] + learning_rate * params_grad[1]
    # delta = (velocity1[i], velocity2[i])
    # print(params)

    # print(params_grad)

    params = params + delta


print(params)

plt.plot(range(epochs), convergence1, '.-')
plt.plot(range(epochs), 1. / convergence2, '.-')
plt.plot(range(epochs), beta, 'g--')
plt.plot(range(epochs), R, 'g--')
plt.show()



