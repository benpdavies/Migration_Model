import statsmodels.api as sm
import numpy as np
import random
from statsmodels.graphics.api import abline_plot
from scipy import stats
from statsmodels import graphics
import matplotlib.pyplot as plt
import gravity_model

def ci(i, n):
    """
    The multinomial fit is a poisson fit
    with `n` more parameters needed to
    implement the normalization constraints:
    i.e. the poisson variables in each of the
    `n` origin location must sum to one.

    This function selects the appropriate
    parameter relative to the normalization
    constant in each origin location.
    """
    c = list(np.zeros(n))
    c[i] = 1.
    return c

def recover_params(locations, populations, OD):


    # # exponential deterrence function
    Y,EXOG = map(np.array, zip(*[ [ xij ,\
                      [1.] + ci(i, nlocs) + [np.log(populations[j]), - locations[i,j]] ]\
         for i,xi in enumerate(OD) for j,xij in enumerate(xi) \
         if i!=j and populations[j]>0.]) )

    # power law deterrence function
    # Y,EXOG = map(np.array, zip(*[ [ xij ,\
    #                   [1.] + ci(i, nlocs) + [np.log(populations[j]), np.log(distances[i,j])] ]\
    #      for i,xi in enumerate(OD) for j,xij in enumerate(xi) \
    #      if i!=j and populations[j]>0.]) )

    poisson_model = sm.GLM(Y, EXOG, family=sm.families.Poisson(link=sm.families.links.log))
    poisson_results = poisson_model.fit()
    # print(poisson_results.summary())

    # Population exponent
    print(poisson_results.params[-2])

    # Parameter det. funct.

    # exponential deterrence function
    # print  1./poisson_results.params[-1]

    # power law deterrence function
    print(1/poisson_results.params[-1])


    # FIGURES
    # yhat = poisson_results.mu
    # fig, ax = plt.subplots()
    # plt.plot(yhat, Y, 'o')
    #
    # xx = np.linspace(min(Y),max(Y),10)
    # plt.plot(xx,xx)
    # # line_fit = sm.OLS(Y, sm.add_constant(yhat, prepend=True)).fit()
    # # abline_plot(model_results=line_fit, ax=ax)
    #
    #
    # ax.set_title('Model Fit Plot')
    # ax.set_ylabel('Observed values')
    # ax.set_xlabel('Fitted values')
    # plt.show()
    # graphics.gofplots.qqplot(poisson_results.resid_pearson, line='r')
    # fig, ax = plt.subplots()

    # plt.plot(yhat, poisson_results.resid_pearson,'o')


    # plt.hlines(0, 0, max(yhat))
    # ax.set_xlim(0, 100)
    # ax.set_title('Residual Dependence Plot')
    # ax.set_ylabel('Pearson Residuals')
    # ax.set_xlabel('Fitted values')
    # plt.show()


locations, populations, OD = gravity_model.main(7, 10000)

recover_params(locations, populations, OD)

