'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


def SimpleCNNDropout(inputShape, noOfClasses=4, optimizer=Adam(), verbose=0):
  # Build a sequential convolutional neural network architecture.
  model = Sequential(
    [  # First convolutional block: 32 filters with 3x3 kernel, stride 2, ReLU activation, and same padding.
      Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", input_shape=inputShape),
      # Max pooling to reduce spatial dimensions.
      MaxPooling2D(pool_size=(2, 2)),
      # Second convolutional block: 64 filters.
      Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
      # Additional pooling.
      MaxPooling2D(pool_size=(2, 2)),
      # Third convolutional block: 128 filters.
      Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
      # Final pooling before classifier.
      MaxPooling2D(pool_size=(2, 2)),
      # Flatten feature maps to a 1D vector for the dense layers.
      Flatten(),
      # Apply dropout for strong regularization.
      Dropout(0.5),
      # Dense hidden layer with 128 units and ReLU activation.
      Dense(128, activation="relu"),
      # Additional dropout to reduce overfitting.
      Dropout(0.5),
      # Dense hidden layer with 64 units and ReLU activation.
      Dense(64, activation="relu"),
      # Output layer for 4-class classification with softmax activation.
      Dense(4, activation="softmax") if (noOfClasses > 2) else Dense(1, activation="sigmoid")
    ])

  # Compile the model with the provided optimizer, categorical crossentropy loss, and common metrics.
  model.compile(
    optimizer=optimizer,
    # You can use "sparse_categorical_crossentropy" if your labels are integers instead of one-hot encoded.
    loss="categorical_crossentropy" if (noOfClasses > 2) else "binary_crossentropy",
    metrics=[
      CategoricalAccuracy() if (noOfClasses > 2) else BinaryAccuracy(),
      Precision(),
      Recall(),
      AUC(),
      TruePositives(name="TP"),
      TrueNegatives(name="TN"),
      FalsePositives(name="FP"),
      FalseNegatives(name="FN"),
    ],
  )

  # Optionally print the model summary when verbose is enabled.
  if (verbose):
    model.summary()

  # Return the compiled Keras model.
  return model


# --------------------------------------------------------------------------- #
# Path to the image you want to classify. Edit this before running the script.
imagePath = r"/path/to/your/image.jpg"
# Path to the trained model you want to use for prediction. Edit this before running the script.
modelDir = r"History/Simple CNN with Dropout"
modelPath = rf"{modelDir}/Model.keras"
# Optional list of class names corresponding to the model's output classes.
# If None, the script will attempt to load from disk or report predictions as indices.
classes = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
# Input shape expected by the model.
inputShape = (256, 256, 3)
# --------------------------------------------------------------------------- #

# Read the image and preprocess it to match the model's expected input format.
img = cv2.imread(imagePath)
if (img is None):
  raise Exception(f"Failed to read image '{imagePath}'. Please check the path and try again.")
# Convert BGR (OpenCV default) to RGB.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Resize to the model's input size.
imgResized = cv2.resize(img, inputShape[:2], interpolation=cv2.INTER_CUBIC)
# Normalize pixel values to [0,1].
imgNorm = imgResized.astype(np.float32) / 255.0
# Add batch dimension for prediction.
x = np.expand_dims(imgNorm, axis=0)

# Load the trained model from the specified path.
if (not os.path.exists(modelPath)):
  raise Exception(f"Model file '{modelPath}' does not exist. Please check the path and try again.")
try:
  model = tf.keras.models.load_model(modelPath)
  print(f"Successfully loaded model from '{modelPath}'.")
except Exception as ex:
  raise Exception(f"Failed to load model from '{modelPath}': {ex}")

# Run prediction on the preprocessed image.
pred = model.predict(x, verbose=0)

# Interpret the prediction based on the model's output shape.
if (pred.shape[-1] == 1):
  # Binary classification case.
  prob = float(pred[0, 0])
  label = 1 if (prob >= 0.5) else 0
  labelName = (classes[1] if classes and len(classes) > 1 else str(label))
  print(f"Predicted: {labelName} (class {label}) with confidence {prob:.4f}.")
else:
  # Multi-class classification case.
  probs = pred[0]
  idx = int(np.argmax(probs))
  conf = float(probs[idx])
  name = (classes[idx] if classes and idx < len(classes) else str(idx))
  print(f"Predicted: {name} (class {idx}) with confidence {conf:.4f}.")

# Plot the input image and the predicted probabilities for each class.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.subplot(1, 2, 2)
if (pred.shape[-1] == 1):
  plt.bar([0, 1], [1 - prob, prob], color=["blue", "orange"])
  plt.xticks(
    [0, 1],
    [classes[0] if classes else "Class 0", classes[1] if classes and len(classes) > 1 else "Class 1"]
  )
  plt.title("Predicted Probabilities")
else:
  plt.bar(range(len(probs)), probs, color="orange")
  plt.xticks(
    range(len(probs)),
    [classes[i] if classes and i < len(classes) else f"Class {i}" for i in range(len(probs))],
    rotation=45
  )
  plt.title("Predicted Probabilities")
plt.tight_layout()
plt.savefig(f"{modelDir}/PredictionResult.png")
plt.show()
plt.close()
