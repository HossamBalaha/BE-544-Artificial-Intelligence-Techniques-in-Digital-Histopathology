# Author: Hossam Magdy Balaha
# Date: June 23th, 2024
# Permissions and Citation: Refer to the README file.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.applications import *
import keras_tuner as kt

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

INPUT_SHAPE = (256, 256, 3)
MAX_EPOCHS = 100


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

# Get the top-5 hyperparameters with the best validation accuracy.
top5BestModels = tuner.get_best_hyperparameters(num_trials=5)

# Get the top-5 results with the best validation accuracy.
top5Results = tuner.oracle.get_best_trials(num_trials=5)

for i, model in enumerate(top5BestModels):
  print(f"Model {i + 1} Hyperparameters:")
  for key, value in model.values.items():
    print(f"\t{key}: {value}")

  print(f"\tValidation Accuracy: {top5Results[i].score}")

# Print the top-5 results summary.
tuner.results_summary(num_trials=5)
