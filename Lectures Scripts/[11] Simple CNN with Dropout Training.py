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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


def SimpleCNN(inputShape, optimizer=Adam(), verbose=0):
  # Build a sequential convolutional neural network architecture.
  model = Sequential([
    # First convolutional block: 32 filters with 3x3 kernel, stride 2, ReLU activation, and same padding.
    Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", input_shape=inputShape),
    # Max pooling to reduce spatial dimensions.
    MaxPooling2D(pool_size=(2, 2)),
    # Second convolutional block: 64 filters.
    Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    # Additional pooling.
    MaxPooling2D(pool_size=(2, 2)),
    # Third convolutional block: 128 filters.
    Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
    # Final pooling before classifier.
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten feature maps to a 1D vector for the dense layers.
    Flatten(),
    # Dense hidden layer with 128 units and ReLU activation.
    Dense(128, activation="relu"),
    # Dense hidden layer with 64 units and ReLU activation.
    Dense(64, activation="relu"),
    # Output layer for 4-class classification with softmax activation.
    Dense(4, activation="softmax"),
  ])

  # Compile the model with the provided optimizer, categorical crossentropy loss, and common metrics.
  model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",  # "sparse_categorical_crossentropy".
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

  # Optionally print the model summary when verbose is enabled.
  if (verbose):
    model.summary()

  # Return the compiled Keras model.
  return model


def SimpleCNNDropout(inputShape, noOfClasses=4, optimizer=Adam(), verbose=0):
  # Build a sequential convolutional neural network architecture.
  model = Sequential(
    [  # First convolutional block: 32 filters with 3x3 kernel, stride 2, ReLU activation, and same padding.
      Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same", input_shape=inputShape),
      # Max pooling to reduce spatial dimensions.
      MaxPooling2D(pool_size=(2, 2)),
      # Second convolutional block: 64 filters.
      Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
      # Additional pooling.
      MaxPooling2D(pool_size=(2, 2)),
      # Third convolutional block: 128 filters.
      Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
      # Final pooling before classifier.
      MaxPooling2D(pool_size=(2, 2)),
      # Flatten feature maps to a 1D vector for the dense layers.
      Flatten(),
      # Apply dropout for strong regularization.
      Dropout(0.5),
      # Dense hidden layer with 128 units and ReLU activation.
      Dense(128, activation="relu"),
      # Additional dropout to reduce overfitting.
      Dropout(0.5),
      # Dense hidden layer with 64 units and ReLU activation.
      Dense(64, activation="relu"),
      # Output layer for 4-class classification with softmax activation.
      Dense(4, activation="softmax") if (noOfClasses > 2) else Dense(1, activation="sigmoid")
    ])

  # Compile the model with the provided optimizer, categorical crossentropy loss, and common metrics.
  model.compile(
    optimizer=optimizer,
    # You can use "sparse_categorical_crossentropy" if your labels are integers instead of one-hot encoded.
    loss="categorical_crossentropy" if (noOfClasses > 2) else "binary_crossentropy",
    metrics=[
      CategoricalAccuracy() if (noOfClasses > 2) else BinaryAccuracy(),
      Precision(),
      Recall(),
      AUC(),
      TruePositives(name="TP"),
      TrueNegatives(name="TN"),
      FalsePositives(name="FP"),
      FalseNegatives(name="FN"),
    ],
  )

  # Optionally print the model summary when verbose is enabled.
  if (verbose):
    model.summary()

  # Return the compiled Keras model.
  return model


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
trainDF, testDF = train_test_split(dataFrame, test_size=0.2, random_state=42, stratify=dataFrame["label"])
# Further split the training set into training and validation sets.
trainDF, valDF = train_test_split(trainDF, test_size=0.25, random_state=42, stratify=trainDF["label"])
print(f"Training set size: {len(trainDF)}")
print(f"Validation set size: {len(valDF)}")
print(f"Testing set size: {len(testDF)}")
# ======================================================================== #

# Define the network input image shape (H, W, C).
inputShape = (256, 256, 3)
# Training batch size.
batchSize = 64
# Number of epochs to train.
epochs = 200
# Define the output directory for saving models and logs.
outputDir = "History/Simple CNN with Dropout"
# Create the output directory if it does not exist.
os.makedirs(outputDir, exist_ok=True)

# Create a data generator for training with simple rescaling.
trainDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1].
)

# Create a generator that reads images and labels from the training DataFrame.
trainGen = trainDataGen.flow_from_dataframe(
  trainDF,  # DataFrame containing training image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize images to match model input size.
  batch_size=batchSize,  # Batch size for training.
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
  batch_size=batchSize,  # Batch size for validation.
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
  batch_size=batchSize,  # Batch size for testing (can be larger since no backprop).
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

# Create the CNN model with dropout using the defined input shape.
model = SimpleCNNDropout(
  inputShape,  # Input shape for the model (height, width, channels).
  optimizer=Adam(),  # Use Adam optimizer for training.
  noOfClasses=noOfClasses,  # Set the number of output classes.
  verbose=1,  # Print model summary when creating the model.
)

# Train the model with several useful callbacks.
history = model.fit(
  trainGen,  # Training data generator.
  epochs=epochs,  # Total number of epochs.
  batch_size=batchSize,  # Batch size (note generators already yield batches).
  validation_data=valGen,  # Validation data generator.
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
  verbose=2,  # Print progress during training (1 for progress bar, 2 for one line per epoch, 0 for silent).
)

# Load the best model saved during training.
model.load_weights(f"{outputDir}/Model.keras")

# Evaluate the trained model on training, validation, and test splits.
for name, dataGen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  # Print which split is being evaluated.
  print(f"{name} Evaluation:")
  # Evaluate generator and capture results.
  result = model.evaluate(dataGen, batch_size=batchSize, verbose=0)
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
yCatPred = model.predict(testGen, batch_size=batchSize, verbose=0)
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
