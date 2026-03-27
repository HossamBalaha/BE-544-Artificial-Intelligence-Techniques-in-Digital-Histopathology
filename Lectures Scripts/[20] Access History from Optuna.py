'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import os, warnings, optuna
import tensorflow as tf

# Ignore warnings.
warnings.filterwarnings("ignore")
# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

whichCategory = "benign"  # Choose between "benign" and "malignant".
whichMagnification = "40X"  # Choose between "40X", "100X", "200X", and "400X".
# Define the output directory for saving models and logs.
optunaStorageDir = f"History/Pretrained Models with Optuna {whichCategory} {whichMagnification}"
# Create the output directory if it does not exist.
os.makedirs(optunaStorageDir, exist_ok=True)
if (not os.path.exists(f"{optunaStorageDir}/PretrainedOptuna.db")):
  raise FileNotFoundError(
    f"Optuna database not found at {optunaStorageDir}/PretrainedOptuna.db."
    f"Please run the hyperparameter optimization script to create the database before accessing it."
  )

# Search for the best hyperparameters.
study = optuna.create_study(
  direction="maximize",  # We want to maximize the categorical accuracy.
  study_name="PretrainedOptuna",  # This is the name of the study.
  load_if_exists=True,  # This will load the study if it exists.
  storage=f"sqlite:///{optunaStorageDir}/PretrainedOptuna.db",  # This is the database file.
)

print(f"Number of Finished Trials: {len(study.trials)}")

# Get the best hyperparameters.
print("Best Trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))

print("Hyperparameters: ")
for key, value in trial.params.items():
  print("\t{}: {}".format(key, value))

# Get the top-5 hyperparameters with the best validation accuracy.
print("Top-5 Trials:")
topTrials = study.trials[:5]

for i, trial in enumerate(topTrials):
  print(f"Trial {i + 1}:")
  print("Value: {}".format(trial.value))
  print("Hyperparameters: ")
  for key, value in trial.params.items():
    print("\t{}: {}".format(key, value))
