# Author: Hossam Magdy Balaha
# Date: June 4th, 2024
# Permissions and Citation: Refer to the README file.

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

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


def SimpleCNN(inputShape, optimizer=Adam(), verbose=0):
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
    Dense(4, activation="softmax"),
  ])

  model.compile(
    optimizer=optimizer,
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

  if (verbose):
    model.summary()

  return model


inputShape = (256, 256, 3)
batchSize = 64
epochs = 200
samplesPerClass = 300

# Load data.
basePath = r"BACH Updated Extracted/ROIs_0_256_256_32_32"

catsPaths = [
  r"Benign",
  r"Carcinoma In Situ",
  r"Carcinoma Invasive",
  r"Normal",
]

# Load images.
X, Y = [], []
for catPath in catsPaths:
  files = os.listdir(os.path.join(basePath, catPath))
  files = np.random.choice(files, samplesPerClass, replace=False)
  for file in files:
    img = cv2.imread(os.path.join(basePath, catPath, file))
    img = cv2.resize(img, inputShape[:2], interpolation=cv2.INTER_CUBIC)
    X.append(img)
    Y.append(catPath)

# Preprocess data.
X = np.array(X) / 255.0
enc = LabelEncoder()
Y = enc.fit_transform(Y)
yCat = to_categorical(Y, num_classes=len(catsPaths))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("yCat shape:", yCat.shape)
print("Classes:", enc.classes_)
print("X Data Type:", X.dtype)
print("Y Data Type:", Y.dtype)

# Create the model.
model = SimpleCNN(inputShape, optimizer=Adam(), verbose=1)

# Split the data.
xTrain, xTest, yCatTrain, yCatTest = train_test_split(
  X, yCat, test_size=0.2, stratify=yCat, random_state=42
)

# Train the model.
os.makedirs("../History", exist_ok=True)
history = model.fit(
  xTrain, yCatTrain,
  epochs=epochs,
  batch_size=batchSize,
  validation_split=0.2,
  callbacks=[
    ModelCheckpoint(
      "History/SimpleCNN.h5", save_best_only=True,
      save_weights_only=False, monitor="val_categorical_accuracy",
      verbose=1,
    ),
    EarlyStopping(patience=50),
    CSVLogger("History/SimpleCNN.log"),
    ReduceLROnPlateau(factor=0.5, patience=10),
    TensorBoard(log_dir="History/SimpleCNN/Logs", histogram_freq=1),
  ],
  verbose=1,
)

# Load the best model.
model.load_weights("History/SimpleCNN.h5")

# Evaluate the model.
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

# Plot the training and validation history.
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
plt.savefig("History/SimpleCNN.png")
plt.show()

# Plot the confusion matrix.
yCatPred = model.predict(xTest, batch_size=batchSize, verbose=0)
yPred = np.argmax(yCatPred, axis=1)
yTrue = np.argmax(yCatTest, axis=1)
cm = confusion_matrix(yTrue, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
disp.plot()
plt.show()
