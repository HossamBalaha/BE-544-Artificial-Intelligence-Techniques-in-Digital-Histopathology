'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import Keras layer building blocks.
from tensorflow.keras.layers import *
# Import Keras model classes (Sequential, Model, etc.).
from tensorflow.keras.models import *
# Import optimizers (Adam, SGD, ...).
from tensorflow.keras.optimizers import *
# Import loss functions.
from tensorflow.keras.losses import *
# Import evaluation metrics.
from tensorflow.keras.metrics import *

# Define the input tensor shape (height, width, channels).
inputShape = (128, 128, 3)

# Build a simple sequential convolutional neural network.
model = Sequential([
  # First convolutional layer: 32 filters, 3x3 kernel, stride 2, ReLU activation, same padding, and set input shape.
  Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", input_shape=inputShape),
  # Reduce spatial resolution by a factor of 2.
  MaxPooling2D(pool_size=(2, 2)),
  # Second convolutional layer: 64 filters.
  Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
  # Further spatial downsampling.
  MaxPooling2D(pool_size=(2, 2)),
  # Third convolutional layer: 128 filters.
  Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
  # Final pooling before the dense classifier.
  MaxPooling2D(pool_size=(2, 2)),
  # Flatten feature maps to a 1D vector for the dense layers.
  Flatten(),
  # Fully connected layer with 128 units.
  Dense(128, activation="relu"),
  # Fully connected layer with 64 units.
  Dense(64, activation="relu"),
  # Output layer for 3-class classification with softmax activation.
  Dense(3, activation="softmax"),
])

# Print a summary of the model architecture.
model.summary()

# Compile the model with optimizer, loss, and evaluation metrics.
model.compile(
  # Use Adam optimizer with default hyperparameters.
  optimizer=Adam(),
  # Use categorical crossentropy for one-hot encoded labels.
  loss="categorical_crossentropy",  # "sparse_categorical_crossentropy",
  # Track common classification metrics during training and evaluation.
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
