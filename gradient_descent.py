import numpy as np
import random
import matplotlib.pyplot as plt
import gravity_model

distances, populations, OD = gravity_model.main(3, 10000)
# probs[i, j] = mass_j**mass_power/dist_ij**-deterrance_power

def OD_likelihood(populations, distances, mass_power, deterrance_power, OD ,exp=False):

    log_like = 0
    nlocs = len(OD)
    totals = np.zeros((nlocs))

    for a in range(nlocs):

        e = np.zeros(nlocs)
        for k in range(nlocs):
            e[k] = np.exp(mass_power*np.log(populations[k]) - distances[a, k]/deterrance_power)
        totals[a] = sum(e)



    for i in range(nlocs):
        print('')
        print(totals[i])
        print('')
        for j in range(nlocs):
            z = np.exp(mass_power*np.log(populations[j] - distances[i, j]/deterrance_power))
            print(z)
            print(z/totals[i])
            print(np.log(z/totals[i]))
            # print(OD[i, j])
            value = OD[i, j] * np.log(z/totals[i])
            # print(value)
            log_like += value

    print(log_like)
    return log_like


def gradient_wrt_deterco(OD, distances, populations, pop_co, deter_co):

    deter_grad = 0
    nlocs = len(OD)
    totals = np.zeros(nlocs)

    for a in range(nlocs):
        e = np.zeros(nlocs)
        for k in range(nlocs):
            zed = (distances[a,k]/deter_co**2) * np.exp(pop_co*np.log(populations[k] - distances[a,k]/deter_co))
            # zed = np.log(populations[k]) * np.exp(pop_co*np.log(populations[k]) - distances[a, k]/deter_co)
            e[k] = zed
        totals[a] = np.log(sum(e))

    for i in range(nlocs):
        for j in range(nlocs):
            value = OD[i, j] * (distances[i,j]/deter_co**2 + totals[j])
            deter_grad += value

    return deter_grad


def gradient_wrt_popco(OD, distances, populations, pop_co, deter_co):

    pop_grad = 0
    nlocs = len(OD)
    totals = np.zeros(nlocs)

    for a in range(nlocs):
        e = np.zeros(nlocs)
        for k in range(nlocs):
            zed = np.log(populations[k]) * np.exp(pop_co*np.log(populations[k]) - distances[a, k]/deter_co)
            e[k] = zed
        totals[a] = np.log(sum(e))

    for i in range(nlocs):
        for j in range(nlocs):
            value = OD[i, j] * (populations[j] - totals[j])
            pop_grad += value

    return pop_grad





def likelihood_grad(populations, distances, pop_co, deter_co):

    pop_grad = gradient_wrt_popco(OD, distances, populations, pop_co, deter_co)
    deter_grad = gradient_wrt_deterco(OD, distances, populations, pop_co, deter_co)
    params_grad = [pop_grad[0], deter_grad]


    return params_grad

# print(len(OD))
params = np.array((2, 2))
learning_rate = 0.0003

# OD_likelihood(populations, distances, 1, 5, OD)

# gradient_wrt_popco(OD, distances, populations, 1,1)
#
# gradient_wrt_deterco(OD, distances, populations, 3, 3)



for i in range(1):
    np.random.shuffle(OD)
    for k in range(len(OD)):
        for l in range(len(OD)):
            if k!= l:
                # population = populations[l % len(OD)]
                # distance = distances[k, l]
                params_grad = likelihood_grad(populations, distances, params[0], params[1])
                delta = (learning_rate * params_grad[0], learning_rate * params_grad[1])
                print(params)
                print(params_grad)
                params = params - delta


print(params)



