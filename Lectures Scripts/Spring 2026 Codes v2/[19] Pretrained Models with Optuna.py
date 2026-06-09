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
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from HMB_Spring_2026_Helpers import PretrainedModelOptunaObjectiveFunction

# Ignore warnings.
warnings.filterwarnings("ignore")
# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# ======================================================================== #
# RELATED TO BreakHist DATASET.
# ======================================================================== #
# Dataset source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
whichCategory = "benign"  # Choose between "benign" and "malignant".
whichMagnification = "40X"  # Choose between "40X", "100X", "200X", and "400X".
classes = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
# Base directory where extracted tiles or ROIs are stored.
# Change this path to match your local setup where the BreaKHis dataset is stored.
basePath = rf"/path/to/BreaKHis_v1/histology_slides/breast/{whichCategory}/SOB"

# Check that the base path exists to avoid silent failures.
if (not os.path.exists(basePath)):
  # Raise a clear exception if the expected dataset folder is missing.
  raise Exception(f"Base path '{basePath}' does not exist.")
print(f"Base path '{basePath}' exists. Proceeding with data loading...")

# Create a dictionary mapping class names to their corresponding directory paths.
classPaths = {cls: os.path.join(basePath, cls) for cls in classes}
# Verify that all class directories exist before proceeding.
for cls, path in classPaths.items():
  if (not os.path.exists(path)):
    raise Exception(f"Class path for '{cls}' does not exist at '{path}'.")
print("All class paths exist. Proceeding with image loading...")

xFiles = []  # List to hold image data.
y = []  # List to hold corresponding labels.
# Loop through each class and read images.
for cls, path in classPaths.items():
  cases = os.listdir(path)  # List all cases in the class directory.
  for case in cases:
    casePath = os.path.join(path, case)  # Full path to the case directory.
    if (os.path.isdir(casePath)):  # Ensure it's a directory before listing files.
      # List all image files in the case directory.
      imageNames = os.listdir(os.path.join(casePath, whichMagnification))
      for imgName in imageNames:
        # Full path to the image file.
        imgPath = os.path.join(casePath, whichMagnification, imgName)
        # Verify that the image file exists before attempting to read it.
        if (not os.path.isfile(imgPath)):
          raise Exception(f"Image file '{imgPath}' does not exist.")
        xFiles.append(imgPath)  # Append image data to the list.
        y.append(cls)  # Append corresponding label to the list.
print(f"Loaded {len(xFiles)} images with corresponding labels.")

# Create a DataFrame to organize image paths and labels for easier handling.
dataFrame = pd.DataFrame({
  "image_path": xFiles,
  "label"     : y,
})
print(f"Created DataFrame with {len(dataFrame)} rows.")
print("DataFrame head:\n", dataFrame.head())

noOfClasses = len(dataFrame["label"].unique())
print(f"Number of unique classes: {noOfClasses}")

# Train-test split of the dataset using sklearn's train_test_split function.
trainDF, testDF = train_test_split(
  dataFrame,  # DataFrame containing image paths and labels.
  test_size=0.2,  # Use 20% of the data for testing and 80% for training.
  random_state=42,  # Set a random state for reproducibility of the split.
  stratify=dataFrame["label"],  # Stratify the split based on the labels to maintain class distribution in both sets.
)
# Further split the training set into training and validation sets.
trainDF, valDF = train_test_split(
  trainDF,  # DataFrame containing training image paths and labels.
  test_size=0.25,  # Use 25% of the training data for validation (which is 20% of the original data).
  random_state=42,  # Set a random state for reproducibility of the split.
  stratify=trainDF["label"],  # Stratify the split based on the labels to maintain class distribution in both sets.
)
print(f"Training set size: {len(trainDF)}")
print(f"Validation set size: {len(valDF)}")
print(f"Testing set size: {len(testDF)}")
# ======================================================================== #

# Define the network input image shape (H, W, C).
inputShape = (256, 256, 3)
# Define the maximum number of epochs for training during hyperparameter tuning.
maxEpochs = 100

# Define the output directory for saving models and logs.
outputDir = f"History/Pretrained Models with Optuna {whichCategory} {whichMagnification}"
# Create the output directory if it does not exist.
os.makedirs(outputDir, exist_ok=True)

# Create a data generator for training with simple rescaling.
trainDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1].
  rotation_range=10,  # Randomly rotate images by up to 10 degrees for augmentation.
  width_shift_range=0.15,  # Randomly shift images horizontally by up to 15% of the width for augmentation.
  height_shift_range=0.15,  # Randomly shift images vertically by up to 15% of the height for augmentation.
  shear_range=0.15,  # Randomly apply shearing transformations to images for augmentation.
  zoom_range=0.15,  # Randomly zoom in on images by up to 15% for augmentation.
  horizontal_flip=True,  # Randomly flip images horizontally for augmentation.
  vertical_flip=False,  # Do not flip images vertically as it may not be appropriate for histopathological images.
  fill_mode="nearest",  # Fill in newly created pixels after transformations with the nearest pixel values.
)

# Create a generator that reads images and labels from the training DataFrame.
trainGen = trainDataGen.flow_from_dataframe(
  trainDF,  # DataFrame containing training image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize images to match model input size.
  class_mode="categorical",  # Use one-hot encoded labels for training.
  shuffle=True,  # Shuffle training data to improve generalization.
)

# Create a data generator for validation with simple rescaling.
valDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1].
)

# Create a generator that reads images and labels from the validation DataFrame.
valGen = valDataGen.flow_from_dataframe(
  valDF,  # DataFrame containing validation image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize to model input size.
  class_mode="categorical",  # Use one-hot encoded labels for validation.
  shuffle=True,  # Shuffle validation data (optional, can be False if order matters for metrics).
)

# Create a data generator for testing with simple rescaling.
testDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Preserve ordering for metrics.
)

# Instantiate the test generator to evaluate final performance.
testGen = testDataGen.flow_from_dataframe(
  testDF,  # DataFrame containing test image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize to model input size.
  class_mode="categorical",  # Use one-hot encoded labels for evaluation.
  shuffle=False,  # Do not shuffle test data to maintain order for metrics like confusion matrix.
)

for text, gen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  # Print the number of samples and classes in each generator for verification.
  print(f"Generator '{gen.directory}' has {gen.samples} samples and {len(gen.class_indices)} classes.")
  # Display a sample figure of the first batch of training images and their corresponding labels.
  xBatch, yBatch = next(gen)  # Get the first batch of images and labels from the training generator.
  plt.figure(figsize=(12, 12))  # Create a figure with a specific size for better visibility.
  # Loop through the first 16 images in the batch (or fewer if batch size is smaller).
  for i in range(min(16, len(xBatch))):
    plt.subplot(4, 4, i + 1)  # Create a subplot for each image in a 4x4 grid.
    plt.imshow(xBatch[i])  # Display the image (already rescaled to [0, 1]).
    # Show the class label as the title (convert one-hot to class index).
    clsIdx = np.argmax(yBatch[i])  # Get the class index from the one-hot encoded label.
    # Get the class name from the index.
    clsName = list(gen.class_indices.keys())[list(gen.class_indices.values()).index(clsIdx)]
    # Set the title of the subplot to the class name.
    plt.title(clsName + f" ({clsIdx})")  # Show class name and index for clarity.
    plt.axis("off")  # Hide axes for cleaner visualization.
  plt.tight_layout()  # Adjust layout to prevent overlap.
  # Save the sample batch figure to the output directory.
  plt.savefig(f"{outputDir}/{text}_SampleBatch.png")  # Save the figure to the History folder.
  # plt.show()  # Uncomment this line if you want to see the plot during execution.
  plt.close()  # Close the figure to free memory.

# Search for the best hyperparameters.
study = optuna.create_study(
  direction="maximize",  # We want to maximize the categorical accuracy.
  study_name="PretrainedOptuna",  # This is the name of the study.
  load_if_exists=True,  # This will load the study if it exists.
  storage=f"sqlite:///{outputDir}/PretrainedOptuna.db",  # This is the database file.
)

# Create an instance of the objective function for Optuna, which will be called for each trial to
# evaluate the model with different hyperparameters.
objFunc = PretrainedModelOptunaObjectiveFunction(
  trainGen=trainGen,  # Training data generator.
  valGen=valGen,  # Validation data generator.
  testGen=testGen,  # Testing data generator for final evaluation.
  inputShape=inputShape,  # Input shape for the model.
  noOfClasses=noOfClasses,  # Number of classes in the dataset.
  maxEpochs=maxEpochs,  # Maximum number of epochs for training.
)

study.optimize(
  objFunc,
  # Number of hyperparameter combinations to try. Adjust this based on your computational resources and
  # time constraints.
  n_trials=10,
  show_progress_bar=True,  # Show a progress bar during the optimization process for better visibility of progress.
  # Garbage collect after each trial to free up memory, especially important when training deep learning models.
  gc_after_trial=True,
  # Set a timeout of 2 hours (7200 seconds) to prevent excessively long runs. Adjust this based on your
  # expected training time and computational resources.
  timeout=7200,
  # Number of parallel jobs to run. Set to 1 for sequential execution. Increase this if you have multiple
  # GPUs or want to speed up the search, but be cautious of memory constraints.
  n_jobs=1,
)

print(f"Number of Finished Trials: {len(study.trials)}")

# Get the best hyperparameters.
print("Best Trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))

print("Hyperparameters: ")
for key, value in trial.params.items():
  print("\t{}: {}".format(key, value))
