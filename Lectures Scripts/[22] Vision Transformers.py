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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from HMB_Spring_2026_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")
# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

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
configs["HiddenActivation"] = "gelu"
configs["OutputActivation"] = "softmax"
configs["NumClasses"] = 4
configs["Epochs"] = 100

model = BasicVisionTransformer(configs)
model.summary()
