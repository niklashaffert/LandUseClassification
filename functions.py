from pickle import dump, load

class DriveStorage:
    @staticmethod
    def store_drive(object, path):
        """Store an object to a file using pickle."""
        with open(path, 'wb') as f:
            dump(object, f)

    @staticmethod
    def read_drive(path):
        """Read an object from a file using pickle."""
        with open(path, 'rb') as f:
            obj = load(f)
        return obj
