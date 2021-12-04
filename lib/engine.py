from .data import load_data
from .model import MultiTargetModel
from .training import train_model
import tensorflow as tf
import os


class Engine:
    """Main class of the library."""

    def load_data(self, path):
        if not os.path.isfile(path):
            raise ValueError(f'File not found: {path}')
        X, y = load_data(path)
        return X, y

    def load_model(self):
        return MultiTargetModel()

    def train_model(self, model, X, y, task2_loss_multiplier=1):
        return train_model(model, X, y, task2_loss_multiplier)

    def print_variables(self, model):
        B1 = tf.squeeze(model.B1.get_weights()[0]).numpy()
        b3 = model.b3.numpy()[0]
        b4 = model.b4.numpy()[0]
        B2 = B1 * b4

        print('Model variables:')
        print('B1:', B1)
        print('B2:', B2)
        print('b3:', b3)
        print('b4:', b4)
