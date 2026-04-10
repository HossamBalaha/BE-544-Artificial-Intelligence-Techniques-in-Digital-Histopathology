'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from HMB_Spring_2026_Helpers import *

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
basePath = rf"C:\Users\Hossam\Downloads\BreaKHis_v1/histology_slides/breast/{whichCategory}/SOB"

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

# CONFIGS 1:
configs = {}
configs["NumEncoderLayers"] = 5
configs["MLPDimension"] = 128
configs["NumAttentionHeads"] = 5
configs["DropoutRatio"] = 0.1
configs["PatchSize"] = 16
configs["BatchSize"] = 32
configs["NumChannels"] = 3
configs["ImageSize"] = (128, 128, configs["NumChannels"])
configs["EmbedDimension"] = configs["PatchSize"] * configs["PatchSize"] * configs["NumChannels"]
configs["NumPatches"] = int(
  (configs["ImageSize"][0] // configs["PatchSize"]) *
  (configs["ImageSize"][1] // configs["PatchSize"])
)
configs["LossFunction"] = "categorical_crossentropy"
configs["LearningRate"] = 0.0001
configs["Optimizer"] = Adam(learning_rate=configs["LearningRate"])
configs["HiddenActivation"] = "relu"
configs["OutputActivation"] = "softmax"
configs["NumClasses"] = 4
configs["Epochs"] = 100

outputDir = f"History/ViT Config1 {whichCategory} {whichMagnification}"
os.makedirs(outputDir, exist_ok=True)

# Store configs as JSON file in the output directory.
# with open(f"{outputDir}/Configs.json", "w") as f:
#   json.dump(configs, f, indent=4)

# Create generators from DataFrames
trainGen = PatchDataGeneratorFromDataFrame(
  dataFrame=trainDF,
  inputShape=configs["ImageSize"],
  batchSize=configs["BatchSize"],
  classMode="categorical",
  noOfPatches=configs["NumPatches"],
  patchSize=configs["PatchSize"],
  embedDimension=configs["EmbedDimension"],
  shuffle=True
)

valGen = PatchDataGeneratorFromDataFrame(
  dataFrame=valDF,
  inputShape=configs["ImageSize"],
  batchSize=configs["BatchSize"],
  classMode="categorical",
  noOfPatches=configs["NumPatches"],
  patchSize=configs["PatchSize"],
  embedDimension=configs["EmbedDimension"],
  shuffle=False
)

testGen = PatchDataGeneratorFromDataFrame(
  dataFrame=testDF,
  inputShape=configs["ImageSize"],
  batchSize=configs["BatchSize"],
  classMode="categorical",
  noOfPatches=configs["NumPatches"],
  patchSize=configs["PatchSize"],
  embedDimension=configs["EmbedDimension"],
  shuffle=False
)

clear_session()
model = BasicVisionTransformer(configs)
# model.summary()

# Train the model.
history = model.fit(
  trainGen,
  epochs=configs["Epochs"],
  batch_size=configs["BatchSize"],
  validation_data=valGen,
  callbacks=[
    # Save the best model (by validation categorical accuracy).
    ModelCheckpoint(
      f"{outputDir}/Model.keras",  # Filepath to save the best model.
      save_best_only=True,  # Only save the model if validation accuracy improves.
      save_weights_only=False,  # Save the entire model (architecture + weights).
      monitor="val_loss",  # Metric to monitor for improvement.
      verbose=1,  # Print messages when a new best model is saved.
    ),
    # Stop training early if validation stops improving.
    EarlyStopping(patience=50),
    # Record training history to CSV for later analysis.
    CSVLogger(f"{outputDir}/Log.log"),
    # Reduce learning rate when a monitored metric plateaus.
    ReduceLROnPlateau(factor=0.5, patience=10),
    # TensorBoard callback to log training for visualization.
    TensorBoard(log_dir=f"{outputDir}/TB/Logs", histogram_freq=1),
  ],
  verbose=1,
)

# Load the best model saved during training.
model.load_weights(f"{outputDir}/Model.keras")

# Evaluate the trained model on training, validation, and test splits.
for name, dataGen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  # Print which split is being evaluated.
  print(f"{name} Evaluation:")
  # Evaluate generator and capture results.
  result = model.evaluate(dataGen, batch_size=configs["BatchSize"], verbose=0)
  # Print the common metrics in the order returned by model.evaluate.
  print("Loss:", result[0])
  print("Accuracy:", result[1])
  print("Precision:", result[2])
  print("Recall:", result[3])
  print("AUC:", result[4])
  print("TP:", result[5])
  print("TN:", result[6])
  print("FP:", result[7])
  print("FN:", result[8])

# Plot training and validation loss and accuracy curves across epochs.
plt.figure()
plt.subplot(2, 1, 1)
# Plot training and validation loss.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2, 1, 2)
# Plot training and validation categorical accuracy.
plt.plot(history.history["categorical_accuracy"], label="Training Accuracy")
plt.plot(history.history["val_categorical_accuracy"], label="Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
# Save the figure to the History folder for later review.
plt.savefig(f"{outputDir}/History.png")
# Display the plot interactively.
# plt.show()  # Uncomment this line if you want to see the plot during execution.
plt.close()

# Generate predictions on the test set using the trained model.
yCatPred = model.predict(testGen, batch_size=configs["BatchSize"], verbose=0)
# Convert the one-hot predictions to class indices.
yPred = np.argmax(yCatPred, axis=1)
# Get true class indices from the test generator.
yTrue = testGen.classes
# Compute the confusion matrix between true and predicted labels.
cm = confusion_matrix(yTrue, yPred)
# Create a display object and plot the confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testGen.class_indices)
# Show the confusion matrix plot.
disp.plot()
# Rotate x-axis labels for better readability.
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap of labels and titles.
# Save the confusion matrix figure to the History folder.
plt.savefig(f"{outputDir}/ConfusionMatrix.png")
# plt.show()  # Uncomment this line if you want to see the plot during execution.
plt.close()

# Calculate all metrics.
print("Confusion Matrix:")
cm = confusion_matrix(yTrue, yPred)
print("Performance Metrics:")
results = CalculateAllMetrics(cm)
for key, value in results.items():
  print(f"{key}:", value)

# CONFIGS 2:
# configs = {}
# configs["NumEncoderLayers"] = 1
# configs["MLPDimension"] = 64
# configs["NumAttentionHeads"] = 1
# configs["DropoutRatio"] = 0.1
# configs["PatchSize"] = 16
# configs["BatchSize"] = 16
# configs["NumChannels"] = 3
# configs["ImageSize"] = (128, 128, configs["NumChannels"])
# configs["EmbedDimension"] = configs["PatchSize"] * configs["PatchSize"] * configs["NumChannels"]
# configs["NumPatches"] = int(
#   (configs["ImageSize"][0] // configs["PatchSize"]) *
#   (configs["ImageSize"][1] // configs["PatchSize"])
# )
# configs["LossFunction"] = "categorical_crossentropy"
# configs["LearningRate"] = 0.0001
# configs["Optimizer"] = Adam(learning_rate=configs["LearningRate"])
# configs["HiddenActivation"] = "gelu"  # Gaussian Error Linear Unit.
# configs["OutputActivation"] = "softmax"
# configs["NumClasses"] = 4
# configs["Epochs"] = 100

# CONFIGS 3:
# configs = {}
# configs["NumEncoderLayers"] = 1
# configs["MLPDimension"] = 64
# configs["NumAttentionHeads"] = 10
# configs["DropoutRatio"] = 0.1
# configs["PatchSize"] = 16
# configs["BatchSize"] = 16
# configs["NumChannels"] = 3
# configs["ImageSize"] = (128, 128, configs["NumChannels"])
# configs["EmbedDimension"] = configs["PatchSize"] * configs["PatchSize"] * configs["NumChannels"]
# configs["NumPatches"] = int(
#   (configs["ImageSize"][0] // configs["PatchSize"]) *
#   (configs["ImageSize"][1] // configs["PatchSize"])
# )
# configs["LossFunction"] = "categorical_crossentropy"
# configs["LearningRate"] = 0.0001
# configs["Optimizer"] = Adam(learning_rate=configs["LearningRate"])
# configs["HiddenActivation"] = "gelu"  # Gaussian Error Linear Unit.
# configs["OutputActivation"] = "softmax"
# configs["NumClasses"] = 4
# configs["Epochs"] = 100

# CONFIGS 4:
# configs = {}
# configs["NumEncoderLayers"] = 1
# configs["MLPDimension"] = 512
# configs["NumAttentionHeads"] = 25
# configs["DropoutRatio"] = 0.25
# configs["PatchSize"] = 16
# configs["BatchSize"] = 16
# configs["NumChannels"] = 3
# configs["ImageSize"] = (128, 128, configs["NumChannels"])
# configs["EmbedDimension"] = configs["PatchSize"] * configs["PatchSize"] * configs["NumChannels"]
# configs["NumPatches"] = int(
#   (configs["ImageSize"][0] // configs["PatchSize"]) *
#   (configs["ImageSize"][1] // configs["PatchSize"])
# )
# configs["LossFunction"] = "categorical_crossentropy"
# configs["LearningRate"] = 0.0001
# configs["Optimizer"] = Adam(learning_rate=configs["LearningRate"])
# configs["HiddenActivation"] = "gelu"  # Gaussian Error Linear Unit.
# configs["OutputActivation"] = "softmax"
# configs["NumClasses"] = 4
# configs["Epochs"] = 100

# # Visualize the data.
# sampleBatch = trainGen[0]
# sampleRecord = sampleBatch[0][0]
# sampleRecordImg = sampleRecord.reshape(
#   configs["NumPatches"], configs["PatchSize"], configs["PatchSize"], configs["NumChannels"]
# )
# sampleRecordImg = sampleRecordImg * 255.0
# sampleRecordImg = np.uint8(sampleRecordImg)
#
# # print(sampleRecordImg.shape)
#
# plt.figure()
# for i in range(configs["NumPatches"][0]):
#   for j in range(configs["NumPatches"][1]):
#     plt.subplot(configs["NumPatches"][0], configs["NumPatches"][1], i * configs["NumPatches"][0] + j + 1)
#     plt.imshow(cv2.cvtColor(sampleRecordImg[i * 8 + j], cv2.COLOR_BGR2RGB))
#     plt.axis("off")
# plt.tight_layout()
# plt.show()
