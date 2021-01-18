# -*- coding: utf-8 -*-
"""Q3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x02X7DPewq3eu_SuIM-e_jJ22yzknUfU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random


# IMPORTING DATASETS
train = pd.read_csv("Datasets/largeTrain.csv").to_numpy()
test = pd.read_csv("Datasets/largeValidation.csv").to_numpy()


# RANDOM SEEDS (for reproducible results)
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# PREPROCESSING
X_train = train[:,1:].astype(np.float32)
y_train = train[:,0]
X_test = test[:,1:].astype(np.float32)
y_test = test[:,0]

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)


# TO ENSURE CORRECT SHAPES
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# DEFINE CUSTOM NN CLASS
class MLP(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes):
        super(MLP, self).__init__()
        # defining the layers and activation fns
        self.linear1 = nn.Linear(n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, n_classes)
        # for random inits
        nn.init.uniform_(self.linear1.weight, a=0.0, b=1.0)
        nn.init.uniform_(self.linear1.bias, a=0.0, b=1.0)
        nn.init.uniform_(self.linear2.weight, a=0.0, b=1.0)
        nn.init.uniform_(self.linear2.bias, a=0.0, b=1.0)
        
    def forward(self, x):
      y = self.linear1(x)
      y = self.relu(y)
      y = self.linear2(y)
      # omit softmax at the end
      return y


# INPUT/OUTPUT layer sizes
n_features = X_train.shape[1]
n_classes = 10

# HIDDEN LAYERS SIZES TO BE TRIED
n_hidden = [4,5,20,50,100,200]

# LEARNING RATES TO BE TRIED
lr=[0.1,0.01,0.001]

# NO. OF ITERATIONS
epochs=100

#INIT THE GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# COPY DATA TO GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)



train_loss_vs_lr=[]
test_loss_vs_lr=[]

for i in range(len(lr)):

  # define model and loss fn
  model = MLP(n_features, n_hidden[0], n_classes) #i=0
  criterion = nn.CrossEntropyLoss()

  # copy model, loss fn to gpu
  model = model.to(device)
  criterion = criterion.to(device)

  optimizer = optim.Adam(model.parameters(), lr=lr[i])  #i=1

  # to collect per epoch loss
  train_loss=[]
  test_loss=[]

  # TRAIN LOOP
  for epoch in range(epochs):
    # predict = forward pass with our model
    y_predicted = model(X_train)

    # print(y_train.shape,y_predicted.shape)

    # loss
    l = criterion(y_predicted, y_train)

    # save train/validation loss for this iter
    with torch.no_grad():
      train_loss.append(l.detach())
      test_loss.append(criterion(model(X_test), y_test))

    # calculate gradients
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

  with torch.no_grad():
    train_loss_vs_lr.append(train_loss[-1])
    test_loss_vs_lr.append(test_loss[-1])

  # LOSS vs ITERS plot
  iters = list(range(epochs))
  plt.clf()
  plt.plot(iters, train_loss, label='train loss')
  plt.plot(iters, test_loss, '--', label='test loss')
  plt.legend()
  plt.suptitle("Learning Rate:"+str(lr[i]))
  plt.savefig("Plots/Q3/lr"+str(lr[i])+".png")

# PLOTTING LOSS VS LEARNING RATE
plt.clf()
plt.plot(lr, train_loss_vs_lr, label='train loss')
plt.plot(lr, test_loss_vs_lr, '--', label='test loss')
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Cross Entropy Loss')
plt.suptitle("Cross Entropy Loss Vs Learning Rate")
plt.savefig("Plots/Q3/ce_vs_lr.png")

print(lr)
print(train_loss_vs_lr)
print(test_loss_vs_lr)




train_loss_vs_hidden=[]
test_loss_vs_hidden=[]

for i in range(len(n_hidden)):
  # define model, loss fn
  model = MLP(n_features, n_hidden[i], n_classes) #i=0
  criterion = nn.CrossEntropyLoss()

  # copy model, loss fn to GPU
  model = model.to(device)
  criterion = criterion.to(device)

  # init the optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr[1])  #i=1

  # for per epoch loss capture
  train_loss=[]
  test_loss=[]

  for epoch in range(epochs):
    # predict = forward pass with our model
    y_predicted = model(X_train)

    # print(y_train.shape,y_predicted.shape)

    # loss
    l = criterion(y_predicted, y_train)

    with torch.no_grad():
      train_loss.append(l.detach())
      test_loss.append(criterion(model(X_test), y_test))
      # if epoch%10==0:
      #   y_pred = model(X_test)
      #   print("Test Acc:",torch.mean((y_pred.argmax(1) == y_test).float()).item())

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()
  
  with torch.no_grad():
    train_loss_vs_hidden.append(train_loss[-1])
    test_loss_vs_hidden.append(test_loss[-1])

  iters = list(range(epochs))
  # plt.figure(figsize=(20,10))
  plt.clf()
  plt.plot(iters, train_loss, label='train loss')
  plt.plot(iters, test_loss, '--', label='test loss')
  plt.legend()
  plt.suptitle("Hidden Units:"+str(n_hidden[i]))
  plt.savefig("Plots/Q3/hidden"+str(n_hidden[i])+".png")

#PLOTTING LOSS VS HIDDEN LAYER UNITS
plt.clf()
plt.plot(n_hidden, train_loss_vs_hidden, label='train loss')
plt.plot(n_hidden, test_loss_vs_hidden, '--', label='test loss')
plt.xlabel('Hidden Layer Units')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.suptitle("Cross Entropy Loss Vs Hidden Layer Units")
plt.savefig("Plots/Q3/ce_vs_hidden.png")

print(n_hidden)
print(train_loss_vs_hidden)
print(test_loss_vs_hidden)



