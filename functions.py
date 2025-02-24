import numpy as np
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
import subprocess
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm'])

import timm

def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)

def read_drive(path):
  with open(path, 'rb') as f:
      obj = load(f)
  return obj

def resnet_extract_patterns_helper(img, preprocess, pattern_extractor):
  img = preprocess(img).unsqueeze(0)
  with torch.no_grad():
    features = pattern_extractor(img)
  return features.squeeze().numpy().flatten()

def resnet_extract_patterns(df, preprocess):
  # Load ResNet-18 model pretrained on ImageNet
  resnet18 = models.resnet18(pretrained=True)

  # Remove the classification layer (fully connected layer)
  pattern_extractor = nn.Sequential(*list(resnet18.children())[:-1])  # Remove last layer
  pattern_extractor.eval()

  tqdm.pandas()

  # Apply pattern extraction to all images
  pattern = df['Image'].progress_apply(resnet_extract_patterns_helper, args=(preprocess, pattern_extractor,))
  return pattern

def swin_vit_extract_patterns_helper(img, preprocess, model):
    img = preprocess(img).unsqueeze(0)  # Apply transformations & add batch dimension
    with torch.no_grad():
        features = model.forward_features(img)  # Forward pass
    return features.squeeze().numpy().flatten()

def swin_vit_extract_patterns(df, preprocess, swin_vit):
  model = timm.create_model(swin_vit, pretrained=True)
  model.eval()

  tqdm.pandas()

  pattern = df['Image'].progress_apply(swin_vit_extract_patterns_helper, args=(preprocess, model,))
  return pattern

def fit_rf(df, col_predictor):
    x = np.vstack(df[col_predictor].values)
    y = df['Label']

    rf = RandomForestClassifier(random_state = np.random.seed(1234))
    rf.fit(x, y)
    return rf

def fit_nn(df, col_predictor, num_epochs = 10):
  torch.manual_seed(1234)
  torch.cuda.manual_seed_all(1234) 
  
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
