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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Print TensorFlow version and available GPUs for information.
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


# Define a function that builds a simple CNN model given an input shape and optimizer.
def SimpleCNN(inputShape, optimizer=Adam(), verbose=0):
  # Build a sequential convolutional neural network architecture.
  model = Sequential([
    # First convolutional block: 32 filters with 3x3 kernel, stride 2, ReLU activation, and same padding.
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
    # Dense hidden layer with 128 units and ReLU activation.
    Dense(128, activation="relu"),
    # Dense hidden layer with 64 units and ReLU activation.
    Dense(64, activation="relu"),
    # Output layer for 4-class classification with softmax activation.
    Dense(4, activation="softmax"),
  ])

  # Compile the model with the provided optimizer, categorical crossentropy loss, and common metrics.
  model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",  # "sparse_categorical_crossentropy".
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

  # Optionally print the model summary when verbose is enabled.
  if (verbose):
    model.summary()

  # Return the compiled Keras model.
  return model


# Define input image shape for the network (height, width, channels).
inputShape = (256, 256, 3)
# Set the training batch size.
batchSize = 64
# Set the number of training epochs.
epochs = 200
# Number of samples to load per class for balanced sampling.
samplesPerClass = 300

# Base path where extracted ROI tiles are stored.
basePath = r"BACH Updated Extracted/ROIs_0_256_256_32_32"

# List of class folder names to load data from.
catsPaths = [
  r"Benign",
  r"Carcinoma In Situ",
  r"Carcinoma Invasive",
  r"Normal",
]

# Initialize lists to hold images and labels.
X, Y = [], []
# Iterate over each category folder and sample images.
for catPath in catsPaths:
  # List files inside the category directory.
  files = os.listdir(os.path.join(basePath, catPath))
  # Randomly choose a fixed number of files per class without replacement.
  files = np.random.choice(files, samplesPerClass, replace=False)
  for file in files:
    # Read the image from disk using OpenCV.
    img = cv2.imread(os.path.join(basePath, catPath, file))
    # Resize the image to the network's expected input dimensions.
    img = cv2.resize(img, inputShape[:2], interpolation=cv2.INTER_CUBIC)
    # Append image and corresponding class label.
    X.append(img)
    Y.append(catPath)

# Convert image list to NumPy array and normalize pixel values to [0, 1].
X = np.array(X) / 255.0
# Initialize a label encoder to transform string labels to integer indices.
enc = LabelEncoder()
# Fit the encoder and transform labels to integer form.
Y = enc.fit_transform(Y)
# Convert integer labels to one-hot encoded vectors for training.
yCat = to_categorical(Y, num_classes=len(catsPaths))

# Print dataset shapes and types for verification.
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("yCat shape:", yCat.shape)
print("Classes:", enc.classes_)
print("X Data Type:", X.dtype)
print("Y Data Type:", Y.dtype)

# Create and compile the model with the specified input shape and optimizer.
model = SimpleCNN(inputShape, optimizer=Adam(), verbose=1)

# Split the dataset into training and test sets with stratification to preserve class distribution.
xTrain, xTest, yCatTrain, yCatTest = train_test_split(
  X, yCat, test_size=0.2, stratify=yCat, random_state=42
)

# Ensure the History directory exists for saving weights and logs.
os.makedirs("../History", exist_ok=True)
# Train the model and capture the history object for plotting.
history = model.fit(
  xTrain, yCatTrain,
  epochs=epochs,
  batch_size=batchSize,
  validation_split=0.2,
  callbacks=[
    # Save the best model by validation categorical accuracy during training.
    ModelCheckpoint(
      "History/SimpleCNN.h5", save_best_only=True,
      save_weights_only=False, monitor="val_categorical_accuracy",
      verbose=1,
    ),
    # Stop training early if validation performance stops improving.
    EarlyStopping(patience=50),
    # Log training progress to a CSV file.
    CSVLogger("History/SimpleCNN.log"),
    # Reduce learning rate when a metric has stopped improving.
    ReduceLROnPlateau(factor=0.5, patience=10),
    # TensorBoard callback for visualizing training.
    TensorBoard(log_dir="History/SimpleCNN/Logs", histogram_freq=1),
  ],
  verbose=1,
)

# Load the best saved model weights from training.
model.load_weights("History/SimpleCNN.h5")

# Evaluate the model on both training and test sets and print metrics.
for (_x, _y) in [(xTrain, yCatTrain), (xTest, yCatTest)]:
  result = model.evaluate(_x, _y, batch_size=batchSize, verbose=0)
  print("Loss:", result[0])
  print("Categorical Accuracy:", result[1])
  print("Precision:", result[2])
  print("Recall:", result[3])
  print("AUC:", result[4])
  print("TP:", result[5])
  print("TN:", result[6])
  print("FP:", result[7])
  print("FN:", result[8])

# Plot training and validation loss and accuracy over epochs.
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2, 1, 2)
plt.plot(history.history["categorical_accuracy"], label="Training Accuracy")
plt.plot(history.history["val_categorical_accuracy"], label="Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
# Save the training figure for reference.
plt.savefig("History/SimpleCNN.png")
plt.show()

# Compute predictions on the test set and display a confusion matrix.
yCatPred = model.predict(xTest, batch_size=batchSize, verbose=0)
yPred = np.argmax(yCatPred, axis=1)
yTrue = np.argmax(yCatTest, axis=1)
cm = confusion_matrix(yTrue, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
disp.plot()
plt.show()
