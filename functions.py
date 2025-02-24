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

# Function to load the model
def load_model(model_path, input_size, num_classes):
    """
    Load a trained model from a file.

    Parameters:
    - model_path: Path to the saved model file.
    - input_size: Number of input features.
    - num_classes: Number of output classes.

    Returns:
    - model: The loaded neural network model.
    """
    # Create a new instance of the model
    model = Net(input_size, num_classes)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def fit_nn(df, col_predictor, num_epochs=10, seed=1234):
    """
    Fit a neural network model and save it to a file.

    Parameters:
    - df: DataFrame containing the data.
    - col_predictor: List of columns to use as predictors.
    - num_epochs: Number of epochs to train.
    - seed: Random seed for reproducibility.
    - model_path: Path to save the trained model.
    """
    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using GPU

    # Convert data to PyTorch tensors
    x = torch.tensor(np.vstack(df[col_predictor].values), dtype=torch.float32)
    y = torch.tensor(df['Label'].values, dtype=torch.long)

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

    return model  # Return the trained model
