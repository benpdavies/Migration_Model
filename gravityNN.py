import torch
import numpy as np
import gravity_model
import random
import matplotlib.pyplot as plt


class GravityModel_MultinomialRegression(torch.nn.Module):

    def __init__(self, dim_features=2):
        super(GravityModel_MultinomialRegression, self).__init__()
        self.linear1 = torch.nn.Linear(dim_features, 1)
        self.relu1 = torch.nn.ReLU()

    def forward(self, X):
        out = self.linear1(X)
        # out = self.relu1(out)
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

epochs = 1000
batches = 1000
parames = []

for epoch in range(epochs):

    X, y = gravity_model.create_training_data(nlocs=batches, max_popn=1000, mass_co=2, deterrance_co=2)

    # print(y)

    for batch in range(batches):

        # X = (nlocs*(nlocs-1))x2 vector with destination % population
        trainData = torch.tensor(X[batch], dtype=torch.float)
        # y = (nlocs*(nlocs-1))x1 vector with total journeys
        trainJourneys = torch.tensor(y[batch], dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted y by passing x to the model
        outputs = model.forward(trainData)

        # Compute loss
        loss = model.loss(outputs, trainJourneys)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

        # print(trainData)
        # print(trainJourneys)
        # print(y_pred)

        params = list(model.parameters())
        # print(params[0][0].detach().numpy())
        parames.append(loss.item())


    if epoch % 1 == 0:
        print(epoch, loss.item())

# print(parames)

# plt.plot(parames)
# plt.show()