'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import os, warnings
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from HMB_Spring_2026_Helpers import PretrainedModelKerasTuner

# Ignore warnings.
warnings.filterwarnings("ignore")
# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# Define the network input image shape (H, W, C).
inputShape = (256, 256, 3)
# Define the maximum number of epochs for training during hyperparameter tuning.
maxEpochs = 100
# Define the number of classes in the classification task.
noOfClasses = 4

# Define the output directory for saving models and logs.
whichCategory = "benign"  # Choose between "benign" and "malignant".
whichMagnification = "40X"  # Choose between "40X", "100X", "200X", and "400X".
ktStorageDir = f"History/Pretrained Models with Keras Tuner {whichCategory} {whichMagnification}"
if (not os.path.exists(ktStorageDir)):
  raise Exception(f"The Keras Tuner storage directory '{ktStorageDir}' does not exist.")

# Create an instance of the custom Keras Tuner for hyperparameter tuning of pretrained models.
tuner = PretrainedModelKerasTuner(
  inputShape=inputShape,
  maxEpochs=maxEpochs,
  noOfClasses=noOfClasses,
  directory=ktStorageDir,
  projectName="PretrainedKerasTuner",
)

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
