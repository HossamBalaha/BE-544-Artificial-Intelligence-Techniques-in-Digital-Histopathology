# Author: Hossam Magdy Balaha
# Date: June 23th, 2024
# Permissions and Citation: Refer to the README file.

import os, splitfolders, optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *
from tensorflow.keras.backend import clear_session

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

INPUT_SHAPE = (128, 128, 3)  # Changed from (256, 256, 3) to (128, 128, 3).
MAX_EPOCHS = 5  # Changed from 100 to 5.

basePath = r"BACH Updated Extracted\ROIs_0_256_256_32_32"
splitFolder = r"BACH Updated Extracted\ROIs_0_256_256_32_32_Split"

if (not os.path.exists(splitFolder)):
  splitfolders.fixed(
    basePath,
    output=splitFolder, seed=42,
    fixed=(2000, 500, 500),
  )


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


def ModelBuilder(baseModelStr, optimizerStr, dropout, applyDropout, learningRate):
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

  baseModel = baseModelCls[baseModelStr](
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

  if (applyDropout):
    layers.insert(4, Dropout(dropout))
    layers.insert(3, Dropout(dropout))

  model = Sequential(layers)

  model.compile(
    optimizer=optimizerCls[optimizerStr](learning_rate=learningRate),
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

  return model


def CreateDataGenerators(
  splitFolder, batchSize, inputShape, classMode="categorical"
):
  trainFolder = os.path.join(splitFolder, "train")
  valFolder = os.path.join(splitFolder, "val")
  testFolder = os.path.join(splitFolder, "test")

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
    trainFolder,
    target_size=inputShape[:2],
    class_mode=classMode,
    shuffle=True,
    batch_size=batchSize,
  )

  valDataGen = ImageDataGenerator(
    rescale=1.0 / 255.0,
  )

  valGen = valDataGen.flow_from_directory(
    valFolder,
    target_size=inputShape[:2],
    class_mode=classMode,
    shuffle=True,
    batch_size=batchSize,
  )

  testDataGen = ImageDataGenerator(
    rescale=1.0 / 255.0,
  )

  testGen = testDataGen.flow_from_directory(
    testFolder,
    target_size=inputShape[:2],
    class_mode=classMode,
    shuffle=False,
    batch_size=batchSize,
  )

  return trainGen, valGen, testGen


def ObjectiveFunction(trial):
  try:
    clear_session()

    baseModelStr = trial.suggest_categorical(
      "baseModel",
      ["MobileNetV2", "InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"]
    )

    optimizerClsStr = trial.suggest_categorical(
      "optimizer",
      ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax", "Nadam"]
    )

    dropoutRatio = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
    batchSize = trial.suggest_categorical("batchSize", [8, 16, 32, 64])
    applyDropout = trial.suggest_categorical("applyDropout", [True, False])
    learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True)

    model = ModelBuilder(baseModelStr, optimizerClsStr, dropoutRatio, applyDropout, learningRate)

    trainGen, valGen, testGen = CreateDataGenerators(
      splitFolder, batchSize, INPUT_SHAPE
    )

    os.makedirs("History", exist_ok=True)

    model.fit(
      trainGen,
      epochs=MAX_EPOCHS,
      validation_data=valGen,
      batch_size=batchSize,
      callbacks=[
        EarlyStopping(patience=3),
        ReduceLROnPlateau(factor=0.5, patience=3),
      ],
      verbose=1,
    )

    result = model.evaluate(testGen, batch_size=batchSize, verbose=0)

    return result[1]  # Return the categorical accuracy.

  except Exception as e:
    return 0.0


# Search for the best hyperparameters.
study = optuna.create_study(
  direction="maximize",  # We want to maximize the categorical accuracy.
  study_name="PretrainedOptuna",  # This is the name of the study.
  load_if_exists=True,  # This will load the study if it exists.
  storage="sqlite:///History/PretrainedOptuna/PretrainedOptuna.db",  # This is the database file.
)
study.optimize(
  ObjectiveFunction,
  n_trials=10,  # This is the number of trials.
  show_progress_bar=True,  # This will show a progress bar.
  gc_after_trial=True,  # This will clear the memory after each trial.
  timeout=7200,  # This is the time limit for each trial.
  n_jobs=1,  # This is the number of parallel jobs.
)

print(f"Number of Finished Trials: {len(study.trials)}")

# Get the best hyperparameters.
print("Best Trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))

print("Hyperparameters: ")
for key, value in trial.params.items():
  print("\t{}: {}".format(key, value))
