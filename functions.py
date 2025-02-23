from numpy import vstack, random
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import timm

def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)

def read_drive(path):
  with open(path, 'rb') as f:
      obj = load(f)
  return obj

def predict_rf(df_train, df_pred, col):
    x = vstack(df_train[col].values)
    y = df_train['Label']

    x_validation = vstack(df_pred[col].values)

    rf = RandomForestClassifier(random_state = random.seed(1234))
    rf.fit(x, y)
    pred = rf.predict(x_validation)
    pred_proba = rf.predict_proba(x_validation)
    return pred, pred_proba

def predict_nn(df_train, df_pred, col):
  # Convert data to PyTorch tensors
  X_train = torch.tensor(np.vstack(df_train[col].values), dtype=torch.float32)
  y_train = torch.tensor(df_train['Label'].values, dtype=torch.long)
  X_val = torch.tensor(np.vstack(df_pred[col].values), dtype=torch.float32)

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
  input_size = X_train.shape[1]
  num_classes = len(df_train['Label'].unique())
  model = Net(input_size, num_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Train the model
  num_epochs = 10  # Adjust as needed
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  # Make predictions on the validation set
  with torch.no_grad():
    outputs = model(X_val)
    _, predicted = torch.max(outputs, 1)
    predictions = predicted.numpy()

  return predictions
