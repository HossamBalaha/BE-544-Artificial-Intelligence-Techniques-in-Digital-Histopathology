# Author: Hossam Magdy Balaha
# Date: May 3rd, 2024

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *

inputShape = (128, 128, 3)

model = Sequential([
  Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", input_shape=inputShape),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(128, activation="relu"),
  Dense(64, activation="relu"),
  Dense(3, activation="softmax"),
])

model.summary()

model.compile(
  optimizer=Adam(),
  loss="categorical_crossentropy",  # "sparse_categorical_crossentropy",
  metrics=[
    CategoricalAccuracy(),
    Precision(),
    Recall(),
    AUC(),
    TruePositives(name="TP"),
    TrueNegatives(name="TN"),
    FalsePositives(name="FP"),
    FalseNegatives(name="FN"),
  ],
)
