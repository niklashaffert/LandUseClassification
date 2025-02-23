from numpy import vstack, random
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier

@staticmethod
def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)

@staticmethod
def read_drive(path):
  with open(path, 'rb') as f:
      obj = load(f)
  return obj

@staticmethod
def predict_rf(df_train, df_pred, col):
    x = vstack(df_train[col].values)
    y = df_train['Label']

    x_validation = vstack(df_pred[col].values)

    rf = RandomForestClassifier(random_state = random.seed(1234))
    rf.fit(x, y)
    pred = rf.predict(x_validation)
    pred_proba = rf.predict_proba(x_validation)
    return pred, pred_proba
