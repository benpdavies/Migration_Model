import torch
import numpy as np
import gravity_model
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable


class GravityModel_MultinomialRegression(torch.nn.Module):

    def __init__(self, dim_features=2):
        super(GravityModel_MultinomialRegression, self).__init__()
        self.linear1 = torch.nn.Linear(dim_features, 1)

    def forward(self, X):
        out = self.linear1(X)
        return out

    def loss(self, out, y):
        lsm = torch.nn.LogSoftmax(dim=0)
        # print(y)
        # print(out)
        # print(lsm(torch.squeeze(out)))
        # print(y * lsm(torch.squeeze(out)))
        # print((y * lsm(torch.squeeze(out))).sum())
        return - (y * lsm(torch.squeeze(out))).sum()

model = GravityModel_MultinomialRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)

epochs = 1000
locations = 200
parames = []

X, y = gravity_model.create_training_data(nlocs=locations, max_popn=1000, mass_co=1, deterrance_co=2)
# shape(X) = [batches, (batches-1), 2]
# shape(Y) = [batches, (batches-1)]

for epoch in range(epochs):

    # show model's parameters
    params = list(model.parameters())
    if epoch % 10 == 0:
        print(params[0][0].detach().numpy())

    # zero the parameter gradients
    optimizer.zero_grad()

    # shuffle order of training data each epoch
    idx = list(range(locations))
    np.random.shuffle(idx)

    # initialise loss
    loss = 0

    for batch in range(locations):

        # print(X[idx[batch]])
        # print(y[idx[batch]])

        # X = (nlocs*(nlocs-1))x2 vector with destination % population
        trainData = torch.tensor(X[idx[batch]], dtype=torch.float)
        # y = (nlocs*(nlocs-1))x1 vector with total journeys
        trainJourneys = torch.tensor(y[idx[batch]], dtype=torch.float)

        # Forward pass: Compute predicted y by passing x to the model
        outputs = model.forward(trainData)

        # Compute loss
        loss += model.loss(outputs, trainJourneys)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    parames.append(params[0][0].detach().numpy()[0])

# print(parames)
#
# plt.plot(parames)
# plt.show()