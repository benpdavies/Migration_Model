import numpy as np
import math
from scipy.optimize import minimize

# def lik(parameters):
#     m = parameters[0]
#     b = parameters[1]
#     sigma = parameters[2]
#
#     for i in np.arange(0, len(x)):
#         y_exp = m * x + b
#
#     L = (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 /
#          (2 * sigma ** 2) * sum((y - y_exp) ** 2))
#     return L
#
# x = np.array([1,2,3,4,5])
# y = np.array([2,5,8,11,14])
# lik_model = minimize(lik, np.array([1,1,1]), method='L-BFGS-B')


# n = 1000
# p1, p2, p3 = 0.4, 0.2, 0.3
# p4 = 1-(p1+p2+p3)
# Y = np.random.multinomial(n , (p1,p2,p3,p4))
# print(Y)
#
# def lik(parameters):
#     x1, x2, x3 = parameters[0], parameters[1], parameters[2]
#     x4 = 1-(x1+x2+x3)
#     y1, y2, y3, y4 = Y[0],Y[1],Y[2],Y[3]
#
#     L = y1*np.log(x1) + y2*np.log(x2) + y3*np.log(x3) + y4*np.log(x4)
#     return L
#
# lik_model = minimize(lik, np.array([0.25, 0.25, 0.25]), method='L-BFGS-B')
#
# print(lik_model)






#
# # loglike = math.log(n,y1) + y1*math.log(x1) + y2*math.log(x2) + y3*math.log(x3) + y4*math.log(x4)
# #
# # def find_loglike(i, Y):
# #
# #     log_like = math.log(n,Y(i)) + Y[0]*math.log(x1) + Y[1]*math.log(x2) + Y[2]*math.log(x3) + Y[3]*math.log(x4)
#
# def lik(parameters, Y, n):
#     x1, x2, x3 = parameters[0],parameters[1],parameters[2]
#     x4 = 1-(x1+x2+x3)
#     y1, y2, y3, y4 = Y[0],Y[1],Y[2],Y[3]
#     for i in np.arange(0, n):
#         coeff = np.log(math.factorial(n)/math)
#     L = + y1*np.log(x1) + y2*np.log(x2) + y3*np.log(x3) + y4*np.log(x4)


