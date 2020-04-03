# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:53:32 2020

@author: Nicholas Guthrie
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# define
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile with opt and loss
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# fit
model.fit(xs, ys, epochs=500)

# predict
print(model.predict([10.0]))

