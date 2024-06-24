# Author: Hossam Magdy Balaha
# Date: June 23th, 2024
# Permissions and Citation: Refer to the README file.

import os, splitfolders
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *
import keras_tuner as kt

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

INPUT_SHAPE = (256, 256, 3)
MAX_EPOCHS = 100

basePath = r"BACH Updated Extracted\ROIs_0_256_256_32_32"
splitFolder = r"BACH Updated Extracted\ROIs_0_256_256_32_32_Split"


def CalculateAllMetrics(cm):
  # Calculate TP, TN, FP, FN.
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = np.sum(cm) - (TP + FP + FN)

  results = {}

  # Using macro averaging.
  precision = np.mean(TP / (TP + FP))
  recall = np.mean(TP / (TP + FN))
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.mean(TN / (TN + FP))

  results["Macro Precision"] = precision
  results["Macro Recall"] = recall
  results["Macro F1"] = f1
  results["Macro Accuracy"] = accuracy
  results["Macro Specificity"] = specificity

  # Using micro averaging.
  precision = np.sum(TP) / np.sum(TP + FP)
  recall = np.sum(TP) / np.sum(TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.sum(TN) / np.sum(TN + FP)

  results["Micro Precision"] = precision
  results["Micro Recall"] = recall
  results["Micro F1"] = f1
  results["Micro Accuracy"] = accuracy
  results["Micro Specificity"] = specificity

  # Using weighted averaging.
  samples = np.sum(cm, axis=1)
  weights = samples / np.sum(cm)

  precision = np.sum(TP / (TP + FP) * weights)
  recall = np.sum(TP / (TP + FN) * weights)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.sum(TN / (TN + FP) * weights)

  results["Weighted Precision"] = precision
  results["Weighted Recall"] = recall
  results["Weighted F1"] = f1
  results["Weighted Accuracy"] = accuracy
  results["Weighted Specificity"] = specificity

  return results


def ModelHyperparamsBuilder():
  hp = kt.HyperParameters()

  hp.Choice("baseModel", ["MobileNetV2", "InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"])
  hp.Choice("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax", "Nadam"])
  hp.Choice("learningRate", [1e-2, 1e-3, 1e-4, 1e-5])
  hp.Choice("batchSize", [8, 16, 32, 64])
  hp.Choice("applyDropout", [True, False])
  hp.Choice("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])

  return hp


def ModelBuilder(hp):
  baseModelCls = {
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "ResNet50"   : ResNet50,
    "VGG16"      : VGG16,
    "VGG19"      : VGG19,
    "Xception"   : Xception,
  }

  optimizerCls = {
    "Adam"    : Adam,
    "RMSprop" : RMSprop,
    "SGD"     : SGD,
    "Adagrad" : Adagrad,
    "Adadelta": Adadelta,
    "Adamax"  : Adamax,
    "Nadam"   : Nadam,
  }

  baseModel = baseModelCls[hp["baseModel"]](
    include_top=False,
    weights="imagenet",
    input_shape=INPUT_SHAPE,
  )

  for layer in baseModel.layers:
    layer.trainable = False

  layers = [
    baseModel,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(4, activation="softmax"),
  ]

  if (hp["applyDropout"]):
    layers.insert(4, Dropout(hp["dropout"]))
    layers.insert(3, Dropout(hp["dropout"]))

  model = Sequential(layers)

  model.compile(
    optimizer=optimizerCls[hp["optimizer"]](learning_rate=hp["learningRate"]),
    loss="categorical_crossentropy",
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

  model.summary()

  return model


def ModelTuner():
  hp = ModelHyperparamsBuilder()

  tuner = kt.Hyperband(
    ModelBuilder,
    hyperparameters=hp,
    objective="val_categorical_accuracy",
    max_epochs=MAX_EPOCHS,  # This is the maximum number of epochs to train the model.
    factor=7,  # This is the downsampling factor for the number of epochs.
    directory="History",
    project_name="PretrainedKerasTuner",
    overwrite=False,
  )

  return tuner


tuner = ModelTuner()

trainDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,
  rotation_range=15,  # Changed from 45 to 15.
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.1,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode="nearest",
)

trainGen = trainDataGen.flow_from_directory(
  f"{splitFolder}/train",
  target_size=INPUT_SHAPE[:2],
  class_mode="categorical",
  shuffle=True,
)

valDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,
)

valGen = valDataGen.flow_from_directory(
  f"{splitFolder}/val",
  target_size=INPUT_SHAPE[:2],
  class_mode="categorical",
  shuffle=True,
)

testDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,
)

testGen = testDataGen.flow_from_directory(
  f"{splitFolder}/test",
  target_size=INPUT_SHAPE[:2],
  class_mode="categorical",
  shuffle=False,
)

# Get the best hyperparameters.
bestHP = tuner.get_best_hyperparameters(1)[0]

print("Best Hyperparameters:")
print(bestHP.values)

# Build the model with the best hyperparameters.
model = ModelBuilder(bestHP)

# Train the best model.
history = model.fit(
  trainGen,
  epochs=MAX_EPOCHS,
  batch_size=bestHP["batchSize"],
  validation_data=valGen,
  callbacks=[
    ModelCheckpoint(
      "History/PretrainedKerasTunerBestHP/Chk.h5", save_best_only=True,
      save_weights_only=False, monitor="val_categorical_accuracy",
      verbose=1,
    ),
    EarlyStopping(patience=0.1 * MAX_EPOCHS),
    CSVLogger("History/PretrainedKerasTunerBestHP/Log.log"),
    ReduceLROnPlateau(factor=0.5, patience=0.05 * MAX_EPOCHS),
    TensorBoard(log_dir="History/PretrainedKerasTunerBestHP/Logs", histogram_freq=1),
  ],
  verbose=1,
)

# Load the best model.
model.load_weights("History/PretrainedKerasTunerBestHP/Chk.h5")

# Evaluate the model.
for name, dataGen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  print(f"{name} Evaluation:")
  result = model.evaluate(dataGen, batch_size=bestHP["batchSize"], verbose=0)
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
plt.savefig("History/PretrainedKerasTunerBestHP/Curves.png")
plt.show()

# Plot the confusion matrix.
yCatPred = model.predict(testGen, batch_size=bestHP["batchSize"], verbose=0)
yPred = np.argmax(yCatPred, axis=1)
yTrue = testGen.classes
cm = confusion_matrix(yTrue, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testGen.class_indices)
disp.plot()
plt.savefig("History/PretrainedKerasTunerBestHP/CM.png")
plt.show()

# Calculate all metrics.
results = CalculateAllMetrics(cm)
for key, value in results.items():
  print(f"{key}:", value)
