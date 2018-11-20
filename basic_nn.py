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
        x[i,j,0] = populations[i]
        x[i,j,1] = new_distances[i,j]


# x1 = np.zeros((np.shape(x)[0])*np.shape(x)[1])
# x2 = np.zeros((np.shape(x)[0])*np.shape(x)[1])
# for k in range(np.shape(x)[0]):
#     for l in range(np.shape(x)[1]):
#         b = k*np.shape(x)[0] + l -k
#         x1[b] = x[k,l,0]
#         x2[b] = x[k,l,1]

# X1 = torch.from_numpy(x1)
# print(X1)
# X2 = torch.from_numpy(x2)
x = x.transpose(0,1,2).reshape(90, x.shape[2])
new_OD = np.ravel(new_OD)

# X = 90x2 vector with destination x population
X = torch.tensor(x,  dtype=torch.float32)
# y = 90x1 vector with total journeys
y = torch.tensor(new_OD, dtype=torch.float32)

# X1 = torch.tensor(x[:,0], dtype=torch.float32)
# X2 = torch.tensor(x[:,1], dtype=torch.float32)

# input1 = torch.randn(128, 20)
# print(input1)
# print(input1[0])

# Bilinear model to sum the destination and population
model = nn.Bilinear(90, 90, 90, bias=True)

# Take the softmax of values to find probability
model2 = nn.Softmax()

# Using a simple mean-squared-error loss function
criterion = torch.nn.MSELoss()

# Using a simple stochastic gradient descent optimisers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 50 Epochs
for epoch in range(50):
    # Forward Propagation
    y_pred = model(X1, X2)
    y_pred = model2(y_pred)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())

    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()




