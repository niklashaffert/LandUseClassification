import numpy as np
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

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
