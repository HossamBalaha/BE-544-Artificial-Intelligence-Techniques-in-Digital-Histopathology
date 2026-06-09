'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import torch, os, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from HMB_Spring_2026_Helpers import (
  PretrainedVisionTransformerDataFrame,
  VisionTransformerInferenceDataFrame
)

# Ignore warnings.
warnings.filterwarnings("ignore")
# Check if GPU is available and set the device accordingly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
  # ======================================================================== #
  # RELATED TO BreakHist DATASET.
  # ======================================================================== #
  # Dataset source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
  whichCategory = "benign"  # Choose between "benign" and "malignant".
  whichMagnification = "40X"  # Choose between "40X", "100X", "200X", and "400X".
  classes = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
  # Base directory where extracted tiles or ROIs are stored.
  # Change this path to match your local setup where the BreaKHis dataset is stored.
  basePath = rf"G:/BreaKHis_v1/histology_slides/breast/{whichCategory}/SOB"

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

  # Train-test split of the dataset using sklearn's `train_test_split` function.
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

  configs = {
    # "google/vit-base-patch16-224-in21k",
    # "google/vit-base-patch16-224",
    # "google/vit-large-patch16-224",
    # "google/vit-base-patch32-384",
    # "google/vit-large-patch32-384",
    "modelName"            : "google/vit-base-patch16-224-in21k",
    # Set to True to apply data augmentation during training, which can help improve model generalization
    # by introducing variability in the training data.
    "applyDataAugmentation": True,
    # Use 15% of the data for testing and 85% for training/validation.
    # Adjust this based on the size of your dataset and the need for a robust evaluation set.
    "testSize"             : 0.15,
    # Number of epochs to train the model. Adjust this based on the size of your dataset and computational resources.
    # More epochs can lead to better performance but may also increase training time and risk of overfitting.
    "numTrainEpochs"       : 25,
    # Set the batch size for training.
    # A larger batch size can speed up training but may require more memory.
    # Adjust this based on your GPU capabilities and the size of your dataset.
    "batchSize"            : 32,
    # Set the learning rate for the optimizer.
    # A smaller learning rate can lead to more stable training but may require more epochs to converge,
    # while a larger learning rate can speed up training but may cause instability.
    # Adjust this based on your dataset and model architecture.
    "learningRate"         : 5e-4,
    # Set to True to enable mixed precision training, which can speed up training and reduce memory usage
    # on compatible hardware (like NVIDIA GPUs with Tensor Cores).
    "fp16"                 : True,
    # Set the number of steps between saving model checkpoints.
    "saveSteps"            : 25,
    # Set the number of steps between logging training metrics to the console or a logging system.
    "loggingSteps"         : 25,
  }

  trainer, valMetrics, labelMapping = PretrainedVisionTransformerDataFrame(
    trainDF=trainDF,  # Training DataFrame.
    valDF=valDF,  # Validation DataFrame.
    testDF=testDF,  # Optional: Test DataFrame for early evaluation.
    modelName=configs["modelName"],
    outputDir="./History/ViT-Base-BreaKHis",
    applyDataAugmentation=configs["applyDataAugmentation"],
    numTrainEpochs=configs["numTrainEpochs"],
    batchSize=configs["batchSize"],
    learningRate=configs["learningRate"],
    fp16=configs["fp16"],
    saveSteps=configs["saveSteps"],
    loggingSteps=configs["loggingSteps"],
  )

  testResults = VisionTransformerInferenceDataFrame(
    testDF=testDF,  # Test DataFrame with "image_path" and "label".
    outputDir="./History/ViT-Base-BreaKHis",
  )


if (__name__ == "__main__"):
  main()
