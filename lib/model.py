import tensorflow as tf
from tensorflow.keras import layers, Model


class MultiTargetModel(Model):
    def __init__(self):
        super().__init__()
        self.B1 = layers.Dense(1, use_bias=False)
        scalar_params = dict(shape=[1], initializer='random_uniform', trainable=True)
        self.b3 = self.add_weight(**scalar_params)
        self.b4 = self.add_weight(**scalar_params)

    def call(self, X_in, **kwargs):
        X, z = tf.split(X_in, [6, 1], 1)
        y2 = self.B1(X)
        y1 = tf.math.sigmoid(y2*self.b4 + z*self.b3)
        res = tf.concat([y1, y2], axis=1)
        return {'y1':res,'y2':res}
