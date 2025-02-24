from numpy import vstack, random
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import torch
import torch.nn as nn

def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)

def read_drive(path):
  with open(path, 'rb') as f:
      obj = load(f)
  return obj

def fit_rf(df, col_predictor):
    x = vstack(df[col_predictor].values)
    y = df['Label']

    rf = RandomForestClassifier(random_state = random.seed(1234))
    rf.fit(x, y)
    return rf

def fit_nn(df, col_predictor, num_epochs = 10):
  # Convert data to PyTorch tensors
  x = torch.tensor(np.vstack(df[col_predictor].values), dtype=torch.float32)
  y = torch.tensor(df['Label'].values, dtype=torch.long)

  # Define a simple neural network
  class Net(nn.Module):
    def __init__(self, input_size, num_classes):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(input_size, 512)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return x

  # Initialize the model, loss function, and optimizer
  input_size = x.shape[1]
  num_classes = len(df['Label'].unique())
  model = Net(input_size, num_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Train the model
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  return outputs
