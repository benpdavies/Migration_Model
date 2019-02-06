import torch
import numpy as np
import gravity_model
import random

# Define parameters used in gravity model
mass_coeff = 2
deter_coeff = 5
nlocs = 300
max_pop = 1000


def create_training_data(nlocs, max_pop, mass_coeff, deter_coeff):
    # Create synthetic data
    locations, distances, populations, OD = gravity_model.main(nlocs, max_pop, mass_coeff, deter_coeff)

    # Create new data without origin-origin journeys
    new_distances = np.zeros((nlocs, nlocs - 1))
    new_OD = np.zeros((nlocs, nlocs - 1))

    for i in range(nlocs):
        line = np.delete(distances[i], i)
        OD_line = np.delete(OD[i], i)
        new_distances[i] = line
        if i <= nlocs - 1:
            new_OD[i] = OD_line

    x = np.zeros((nlocs, nlocs - 1, 2))
    # journeys = np.zeros(nlocs, nlocs - 1)

    for i in range(nlocs):
        for j in range(nlocs - 1):
            # journeys[i, j] = new_OD[i, j]
            x[i, j, 0] = populations[j]
            x[i, j, 1] = new_distances[i, j]

    return x, new_OD

class GravityModel_MultinomialRegression(torch.nn.Module):

    def __init__(self, dim_features=2):
        super(GravityModel_MultinomialRegression, self).__init__()
        self.linear1 = torch.nn.Linear(dim_features, 1)

    def forward(self, X):
        out = self.linear1(X)
        return out

    def loss(out, y):
        lsm = torch.nn.LogSoftmax(dim=0)
        return - (y * lsm(torch.squeeze(out))).sum()

def getBatch(X, y):

    idx = random.randint(0, nlocs-1)

    dist_pop = X[idx]
    journeys = y[idx]

    # X = (nlocs*(nlocs-1))x2 vector with destination % population
    xtrain = torch.tensor(dist_pop, dtype=torch.float)
    # y = (nlocs*(nlocs-1))x1 vector with total journeys
    ytrain = torch.tensor(journeys, dtype=torch.float)

    return xtrain, ytrain


model = GravityModel_MultinomialRegression()

# criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

X, y = create_training_data(nlocs, max_pop, mass_coeff, deter_coeff)
#
for t in range(1000):

    # Zero gradients
    optimizer.zero_grad()

    xtrain, ytrain = getBatch(X, y)

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model.forward(xtrain)
    # print(y_pred)

    # Compute and print loss
    loss = GravityModel_MultinomialRegression.loss(y_pred, ytrain)
    if t % 100 == 0:
        # print(y_pred)
        print(t, loss.item())

    # Backward pass
    loss.backward()
    # Update weights
    optimizer.step()