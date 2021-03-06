import torch
import torch.nn as nn
import numpy as np
import gravity_model

# Define parameters used in gravity model
mass_coeff = 1
deter_coeff = 5
nlocs = 10
max_pop = 1000

# Create synthetic data
locations, distances, populations, OD = gravity_model.main(nlocs, max_pop, mass_coeff, deter_coeff)

# Create new data without origin-origin journeys
new_distances = np.zeros((nlocs, nlocs-1))
new_OD = np.zeros((nlocs, nlocs-1))

for i in range(nlocs):
    line = np.delete(distances[i], i)
    OD_line = np.delete(OD[i], i)
    new_distances[i] = line
    if i <= nlocs-1:
        new_OD[i] = OD_line

x = np.zeros((nlocs, nlocs-1, 2))

for i in range(nlocs):
    for j in range(nlocs-1):
        x[i,j,0] = populations[j]
        x[i,j,1] = new_distances[i,j]


# x = x.transpose(0,1,2).reshape(90, x.shape[2])
# new_OD = np.ravel(new_OD)

# X = 10x9x2 vector with destination x population
X = torch.tensor(x, dtype=torch.float)
# y = x1 vector with total journeys
y = torch.tensor(new_OD, dtype=torch.float)


class GravityModel_MultinomialRegression(torch.nn.Module):

    def __init__(self, dim_features=2):
        super(GravityModel_MultinomialRegression, self).__init__()
        self.linear1 = torch.nn.Linear(dim_features, 1)

    def forward(self, X):
        out = self.linear1(X)
        return out


    def loss(out, y):
        lsm = torch.nn.LogSoftmax(dim=0)
        return - ( y * lsm(torch.squeeze(out)) ).sum()

model = GravityModel_MultinomialRegression()

# criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for t in range(100000):

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X)

    # Compute and print loss
    loss = GravityModel_MultinomialRegression.loss(y_pred, y)
    if t % 1000 == 0:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#
# print(model.linear1.weight.data.norm())
# print(y_pred)
# print(y)