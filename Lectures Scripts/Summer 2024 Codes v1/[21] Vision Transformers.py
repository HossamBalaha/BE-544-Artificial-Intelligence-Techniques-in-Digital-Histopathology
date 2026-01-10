# Author: Hossam Magdy Balaha
# Date: June 27th, 2024
# Permissions and Citation: Refer to the README file.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


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

model = VisionTransformer(configs)
model.summary()
