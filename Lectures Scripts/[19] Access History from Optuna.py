# Author: Hossam Magdy Balaha
# Date: June 24th, 2024
# Permissions and Citation: Refer to the README file.

import optuna
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# Search for the best hyperparameters.
study = optuna.create_study(
  direction="maximize",  # We want to maximize the categorical accuracy.
  study_name="PretrainedOptuna",  # This is the name of the study.
  load_if_exists=True,  # This will load the study if it exists.
  storage="sqlite:///History/PretrainedOptuna/PretrainedOptuna.db",  # This is the database file.
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
