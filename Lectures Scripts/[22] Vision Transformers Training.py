# Author: Hossam Magdy Balaha
# Date: June 27th, 2024
# Permissions and Citation: Refer to the README file.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import patchify, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

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


def ImageToPatches(image, noOfPatches, patchSize):
  patches = patchify.patchify(image, (patchSize, patchSize, 3), step=patchSize)
  patches = patches.reshape(-1, patchSize, patchSize, 3)
  patches = patches[:noOfPatches]
  patches = patches.reshape(-1, patchSize * patchSize * 3)
  return patches


class PatchDataGenerator(Sequence):

  def __init__(
    self, folder, inputShape, batchSize, classMode="categorical",
    noOfPatches=256, patchSize=16, embedDimension=768, shuffle=False,
  ):
    self.folder = folder
    self.inputShape = inputShape
    self.batchSize = batchSize
    self.classMode = classMode
    self.noOfPatches = noOfPatches
    self.patchSize = patchSize
    self.embedDimension = embedDimension
    self.classes = os.listdir(folder)
    self.shuffle = shuffle

    self.listOfImages = []
    self.listOfLabels = []

    for label in os.listdir(folder):
      labelFolder = os.path.join(folder, label)
      for image in os.listdir(labelFolder):
        self.listOfImages.append(os.path.join(labelFolder, image))
        self.listOfLabels.append(label)

    self.numImages = len(self.listOfImages)
    self.indices = np.arange(self.numImages)
    self.classIndices = {label: index for index, label in enumerate(self.classes)}

    np.random.shuffle(self.indices)

  def __len__(self):
    return self.numImages // self.batchSize

  def __getitem__(self, index):
    indices = self.indices[index * self.batchSize: (index + 1) * self.batchSize]
    images = np.zeros((self.batchSize, self.noOfPatches, self.embedDimension))
    if (self.classMode == "categorical"):
      labels = np.zeros((self.batchSize, len(self.classIndices)))
    else:
      labels = np.zeros((self.batchSize, 1))

    for i, index in enumerate(indices):
      image = self.listOfImages[index]
      label = self.listOfLabels[index]

      image = cv2.imread(image)
      image = cv2.resize(image, (self.inputShape[1], self.inputShape[0]), interpolation=cv2.INTER_CUBIC)
      patches = ImageToPatches(image, self.noOfPatches, self.patchSize)

      images[i] = patches

      if (self.classMode == "categorical"):
        labels[i, self.classIndices[label]] = 1
      else:
        labels[i] = self.classIndices[label]

    return images / 255.0, labels

  def on_epoch_end(self):
    if (self.shuffle):
      np.random.shuffle(self.indices)

  def __iter__(self):
    for index in range(0, len(self)):
      yield self.__getitem__(index)

  def __next__(self):
    return self.__iter__()


def CreatePatchDataGenerators(folder, inputShape, batchSize, **kwargs):
  trainFolder = os.path.join(folder, "train")
  valFolder = os.path.join(folder, "val")
  testFolder = os.path.join(folder, "test")

  trainGen = PatchDataGenerator(trainFolder, inputShape, batchSize, shuffle=True, **kwargs)
  valGen = PatchDataGenerator(valFolder, inputShape, batchSize, **kwargs)
  testGen = PatchDataGenerator(testFolder, inputShape, batchSize, **kwargs)

  return trainGen, valGen, testGen


class ClassToken(Layer):
  def __init__(self):
    super().__init__()

  def build(self, input_shape):
    wInit = tf.random_normal_initializer()
    self.w = tf.Variable(
      initial_value=wInit(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
      trainable=True
    )

  def call(self, inputs):
    batchSize = tf.shape(inputs)[0]
    hiddenDim = self.w.shape[-1]

    cls = tf.broadcast_to(self.w, [batchSize, 1, hiddenDim])
    cls = tf.cast(cls, dtype=inputs.dtype)
    return cls


def MLP(z, configs):
  z = Dense(configs["MLPDimension"], activation=configs["HiddenActivation"])(z)
  z = Dropout(configs["DropoutRatio"])(z)
  z = Dense(configs["EmbedDimension"])(z)
  z = Dropout(configs["DropoutRatio"])(z)
  return z


def TransformerEncoder(z, configs):
  skipCon = z
  z = LayerNormalization()(z)
  # https://keras.io/api/layers/attention_layers/multi_head_attention/
  z = MultiHeadAttention(
    num_heads=configs["NumAttentionHeads"],  # Number of attention heads.
    key_dim=configs["EmbedDimension"],  # Size of each attention head for query and key.
    value_dim=configs["EmbedDimension"],  # Size of each attention head for value.
    dropout=configs["DropoutRatio"],  # Dropout rate after attention.
  )(z, z)
  z = Add()([z, skipCon])

  skipCon = z
  z = LayerNormalization()(z)
  z = MLP(z, configs)  # Feed Forward Network.
  z = Add()([z, skipCon])

  return z


def PatchEmbedding(z, configs):
  z = Reshape(target_shape=(configs["NumPatches"], configs["EmbedDimension"]))(z)
  z = Dense(configs["EmbedDimension"])(z)
  return z


def PositionEmbedding(configs):
  positions = tf.range(start=0, limit=configs["NumPatches"], delta=1)
  posEmbed = Embedding(
    input_dim=configs["NumPatches"],
    output_dim=configs["EmbedDimension"],
  )(positions)
  return posEmbed


def ClassficationHead(z, configs):
  z = Dense(configs["EmbedDimension"], activation=configs["HiddenActivation"])(z)
  z = Dropout(configs["DropoutRatio"])(z)
  z = Dense(configs["NumClasses"], activation=configs["OutputActivation"])(z)
  return z


def VisionTransformer(configs):
  inputShape = (configs["NumPatches"], configs["EmbedDimension"])
  inputs = Input(inputShape)

  patchEmbed = PatchEmbedding(inputs, configs)  # Create the patch embeddings.
  posEmbed = PositionEmbedding(configs)  # Create the position embeddings.
  Z = patchEmbed + posEmbed  # Add the patch and position embeddings.

  token = ClassToken()(Z)  # Create the class token.
  Z = Concatenate(axis=1)([token, Z])  # Prepend the class token to the patch embeddings.

  # Create the transformer encoder layers.
  for _ in range(configs["NumEncoderLayers"]):
    Z = TransformerEncoder(Z, configs)

  # Final normalization layer.
  Z = LayerNormalization()(Z)

  # Extract the class token.
  cls = Z[:, 0, :]

  # Create the classification head.
  output = ClassficationHead(cls, configs)

  model = Model(inputs, output)

  model.compile(
    optimizer=configs["Optimizer"],
    loss=configs["LossFunction"],
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

clear_session()
model = VisionTransformer(configs)
model.summary()

trainGen, valGen, testGen = CreatePatchDataGenerators(
  splitFolder,
  inputShape=configs["ImageSize"],
  batchSize=configs["BatchSize"],
  classMode="categorical",
  noOfPatches=configs["NumPatches"],
  patchSize=configs["PatchSize"],
  embedDimension=configs["EmbedDimension"],
)

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

# Train the model.
history = model.fit(
  trainGen,
  epochs=configs["Epochs"],
  batch_size=configs["BatchSize"],
  validation_data=valGen,
  callbacks=[
    ModelCheckpoint(
      "History/VisionTransformer/Chk.h5", save_best_only=True,
      save_weights_only=False, monitor="val_categorical_accuracy",
      verbose=1,
    ),
    EarlyStopping(patience=0.25 * configs["Epochs"]),
    CSVLogger("History/VisionTransformer/Log.log"),
    ReduceLROnPlateau(factor=0.5, patience=0.1 * configs["Epochs"]),
    TensorBoard(log_dir="History/VisionTransformer/Logs", histogram_freq=1),
  ],
  verbose=1,
)

# Load the best model.
model.load_weights("History/VisionTransformer/Chk.h5")

# Evaluate the model.
for name, dataGen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  print(f"{name} Evaluation:")
  result = model.evaluate(dataGen, batch_size=configs["BatchSize"], verbose=0)
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
plt.savefig("History/VisionTransformer/Curves.png")
plt.show()

# Plot the confusion matrix.
yCatPred = model.predict(testGen, batch_size=configs["BatchSize"], verbose=0)
yPred = np.argmax(yCatPred, axis=1)
yTrue = testGen.classIndices
cm = confusion_matrix(yTrue, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testGen.class_indices)
disp.plot()
plt.savefig("History/VisionTransformer/CM.png")
plt.show()

# Calculate all metrics.
results = CalculateAllMetrics(cm)
for key, value in results.items():
  print(f"{key}:", value)
