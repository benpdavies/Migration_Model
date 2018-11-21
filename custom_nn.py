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
distances, populations, OD = gravity_model.main(nlocs, max_pop, mass_coeff, deter_coeff)

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


x = x.transpose(0,1,2).reshape(90, x.shape[2])
new_OD = np.ravel(new_OD)

# X = 90x2 vector with destination x population
X = torch.tensor(x, dtype=torch.float)
# y = 90x1 vector with total journeys
y = torch.tensor(new_OD, dtype=torch.float)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in=90):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in)
        self.linear2 = torch.nn.Linear(D_in, D_in, bias=True)
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        sum = self.linear1(X[:,0]) + self.linear2(X[:,1])
        # sum = self.linear1(X)
        y_pred = self.softmax(sum)
        return y_pred

model = TwoLayerNet()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X)

    # Compute and print loss
    loss = criterion(y_pred, y)
    # print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model.linear1.weight.data.norm())