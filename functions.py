@staticmethod
def store_drive(object, path):
  with open(path, 'wb') as f:
      dump(object, f)

@staticmethod
def read_drive(path):
  with open(path, 'rb') as f:
      obj = load(f)
  return obj
     
