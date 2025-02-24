import numpy as np
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torchvision.models as models
import timm
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

def negative_log_likelihood(y_true, y_pred):
    # Ensure predictions are clipped to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    nll = -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
    return nll

def assess_prediction(df, col_pred, model):
  # Compute the accuracy
  accuracy = accuracy_score(df['Label'], df[col_pred])

  # Get the confusion matrix
  cm = confusion_matrix(df['Label'], df[col_pred])

  # Create a mapping from Label -> Name
  label_to_name = dict(zip(df['Label'], df['ClassName']))

  # Sort class names based on unique labels
  class_names = [label_to_name[label] for label in sorted(df['Label'].unique())]

  # Display confusion matrix with correct order
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(cmap='viridis')

  xticks(rotation=90)
  title(f'{model}: Accuracy {accuracy:.3f}')
  show()
