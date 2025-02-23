@staticmethod
def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)
