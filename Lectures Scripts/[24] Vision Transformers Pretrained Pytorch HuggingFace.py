'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import torch, os, tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import *
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import (
  Compose,
  RandomResizedCrop,
  RandomHorizontalFlip,
  ToTensor,
  Normalize,
  Resize,
  CenterCrop
)
from transformers import (
  ViTForImageClassification,
  ViTImageProcessor,
  TrainingArguments,
  Trainer,
  AutoFeatureExtractor,
  AutoModelForImageClassification,
)
from HMB_Spring_2026_Helpers import CalculateAllMetrics

def PretrainedVisionTransformer(
  datasetDir, modelName, outputDir, applyDataAugmentation=True, testSize=0.15,
  numTrainEpochs=32, batchSize=16, learningRate=2e-4, fp16=True, saveSteps=25,
  loggingSteps=10,
):
  # Define the preprocessing functions.
  def _PreprocessTrain(exampleBatch):
    # Convert the images to RGB and apply the transformations.
    # pixel_values is the tensor that the model expects.
    exampleBatch["pixel_values"] = [
      trainTransforms(image.convert("RGB")) for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _PreprocessVal(exampleBatch):
    # Convert the images to RGB and apply the transformations.
    # pixel_values is the tensor that the model expects.
    exampleBatch["pixel_values"] = [
      valTransforms(image.convert("RGB")) for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _CollateFunc(batch):
    # Stack the pixel values and convert the labels to tensors.
    pixelValues = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])

    # pixel_values and labels are the keys that the model expects.
    return {
      "pixel_values": pixelValues,
      "labels"      : labels,
    }

  def _ComputeMetrics(evalPred):
    # Calculate the confusion matrix and the metrics.
    predictions = np.argmax(evalPred.predictions, axis=1)
    references = evalPred.label_ids
    cm = confusion_matrix(references, predictions)
    metrics = CalculateAllMetrics(cm)
    return metrics

  # Load the feature extractor and define the normalization.
  featureExtractor = ViTImageProcessor.from_pretrained(modelName)
  feSize = list(featureExtractor.size.values())
  normalize = Normalize(mean=featureExtractor.image_mean, std=featureExtractor.image_std)

  # Define the transformations for training and validation.
  if (applyDataAugmentation):
    trainTransforms = Compose(
      [
        RandomResizedCrop(feSize),  # Apply random resized crop.
        RandomHorizontalFlip(),  # Apply random horizontal flip.
        ToTensor(),  # Convert to tensor.
        normalize,  # Normalize the image.
      ]
    )
  else:
    trainTransforms = Compose(
      [
        Resize(feSize),  # Resize the image.
        CenterCrop(feSize),  # Center crop the image.
        ToTensor(),  # Convert to tensor.
        normalize,  # Normalize the image.
      ]
    )

  valTransforms = Compose(
    [
      Resize(feSize),  # Resize the image.
      CenterCrop(feSize),  # Center crop the image.
      ToTensor(),  # Convert to tensor.
      normalize,  # Normalize the image.
    ]
  )

  # Load the dataset and get the training subset.
  ds = load_dataset("imagefolder", data_dir=datasetDir)
  ds = ds["train"]
  print("DS:", ds)

  # Split the dataset into training and validation.
  data = ds.train_test_split(test_size=testSize)
  trainDS = data["train"]
  valDS = data["val"]

  print("Train:", trainDS)
  print("Val:", valDS)

  # Apply the transformations to the training and validation datasets.
  trainDS.set_transform(_PreprocessTrain)
  valDS.set_transform(_PreprocessVal)

  # Define the labels.
  labels = data["train"].features["label"].names

  # Define the label mappings.
  label2ID, id2Label = dict(), dict()

  # Create the label mappings.
  for i, label in enumerate(labels):
    label2ID[label] = i  # Update the label to ID mapping.
    id2Label[i] = label  # Update the ID to label mapping.

  # Load the model and define the training arguments.
  model = ViTForImageClassification.from_pretrained(
    modelName,  # Load the model.
    num_labels=len(labels),  # Define the number of labels.
    id2label={str(i): c for i, c in enumerate(labels)},  # Define the ID to label mapping.
    label2id={c: str(i) for i, c in enumerate(labels)},  # Define the label to ID mapping.
    ignore_mismatched_sizes=True,  # Ignore mismatched sizes.
  )

  # Define the training arguments.
  trainingArgs = TrainingArguments(
    output_dir=outputDir,  # Define the output directory.
    per_device_train_batch_size=batchSize,  # Define the batch size.
    evaluation_strategy="steps",  # Define the evaluation strategy.
    num_train_epochs=numTrainEpochs,  # Define the number of training epochs.
    fp16=fp16,  # Define the mixed precision training.
    save_steps=saveSteps,  # Define the save steps.
    eval_steps=saveSteps,  # Define the evaluation steps.
    logging_steps=loggingSteps,  # Define the logging steps.
    learning_rate=learningRate,  # Define the learning rate.
    save_total_limit=2,  # Define the total number of checkpoints to save.
    remove_unused_columns=False,  # Remove unused columns.
    push_to_hub=False,  # Push to the hub.
    report_to="tensorboard",  # Report to tensorboard.
    load_best_model_at_end=True,  # Load the best model at the end.
    log_level="error",  # Define the log level.
  )

  # Create the trainer and train the model.
  trainer = Trainer(
    model,  # Define the model.
    trainingArgs,  # Define the training arguments.
    train_dataset=trainDS,  # Define the training dataset.
    eval_dataset=valDS,  # Define the evaluation dataset.
    tokenizer=featureExtractor,  # Define the tokenizer.
    compute_metrics=_ComputeMetrics,  # Define the compute metrics function.
    data_collator=_CollateFunc,  # Define the data collator.
  )

  # Train the model.
  trainResults = trainer.train()

  # Save the model and the metrics.
  trainer.save_model()

  # Log and save the metrics.
  trainer.log_metrics("train", trainResults.metrics)
  trainer.save_metrics("train", trainResults.metrics)

  # Evaluate the model and save the metrics.
  trainer.save_state()
  metrics = trainer.evaluate()
  trainer.log_metrics("eval", metrics)
  trainer.save_metrics("eval", metrics)

  # Clear the cache to avoid memory issues.
  torch.cuda.empty_cache()


def VisionTransformerInference(imagesBasePath, outputDir, splitType="test"):
  # Load the feature extractor and the model.
  classes = os.listdir(os.path.join(imagesBasePath, splitType))
  featureExtractorX = AutoFeatureExtractor.from_pretrained(outputDir)
  modelX = AutoModelForImageClassification.from_pretrained(outputDir)

  results = []

  # no_grad means that the gradients are not calculated.
  with torch.no_grad():
    # Iterate over the classes.
    for cls in classes:
      # Get the class path and the files.
      clsPath = os.path.join(imagesBasePath, splitType, cls)
      files = os.listdir(clsPath)

      # Iterate over the files.
      for i in tqdm.tqdm(range(len(files))):
        # Get the image path and the extension.
        imagePath = os.path.join(clsPath, files[i])
        extension = imagePath.split(".")[-1]
        if (extension not in ["png", "jpg", "bmp", "jpeg", "tiff", "tif"]):
          continue

        # Open the image and extract the features.
        image = Image.open(imagePath).convert('RGB')
        features = featureExtractorX(image, return_tensors="pt")

        # Apply the model and get the logits.
        outputs = modelX(**features)
        logits = outputs.logits

        # Get the probabilities and the predicted class.
        # softmax(-1) as the probabilities are in the last dimension.
        prob = logits.softmax(-1).max().item()
        probabilities = logits.softmax(-1).tolist()[0]
        predictedClassIDx = logits.argmax(-1).item()

        # Get the predicted class.
        predictedCls = modelX.config.id2label[predictedClassIDx]

        # Store the results.
        recordToStore = {
          "Image Name"        : files[i],
          "Actual Class"      : cls,
          "Predicted Class ID": predictedClassIDx,
          "Predicted Class"   : predictedCls,
          "Probability"       : prob,
          "Probabilities"     : probabilities,
        }
        results.append(recordToStore)

  # Save the results.
  df = pd.DataFrame.from_dict(results)
  df.to_csv(os.path.join(outputDir, f"{splitType.capitalize()}_Results.csv"), index=False)

  references = df["Actual Class"]
  predictions = df["Predicted Class"]

  cm = confusion_matrix(references, predictions)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  disp.plot()
  plt.savefig(os.path.join(outputDir, f"{splitType.capitalize()}_ConfusionMatrix.png"))
  plt.show()
  plt.close()

  # Calculate all metrics.
  results = CalculateAllMetricsUpdated(cm)
  df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
  df.to_csv(os.path.join(outputDir, f"{splitType.capitalize()}_Metrics.csv"))

  # Clear the cache to avoid memory issues.
  torch.cuda.empty_cache()


configs = {
  "applyDataAugmentation": True,
  "testSize"             : 0.15,
  "numTrainEpochs"       : 5,
  "batchSize"            : 32,
  "learningRate"         : 5e-4,
  "fp16"                 : True,
  "saveSteps"            : 250,
  "loggingSteps"         : 100,
}

modelNames = [
  "google/vit-base-patch16-224-in21k",
  # "google/vit-base-patch16-224",
  # "google/vit-large-patch16-224",
  # "google/vit-base-patch32-384",
  # "google/vit-large-patch32-384",
]

bs = configs["batchSize"]
aug = "Aug" if configs["applyDataAugmentation"] else "NoAug"
outputDirs = [
  rf"./History/ViT-Base-{bs}-{aug}-P16-224-In21K",
  # rf"./History/ViT-Base-{bs}-{aug}-P16-224",
  # rf"./History/ViT-Large-{bs}-{aug}-P16-224",
  # rf"./History/ViT-Base-{bs}-{aug}-P32-384",
  # rf"./History/ViT-Large-{bs}-{aug}-P32-384",
]

basePath = r"BACH Updated Extracted\ROIs_0_256_256_32_32"
splitFolder = r"BACH Updated Extracted\ROIs_0_256_256_32_32_Split"

for (modelName, outputDir) in zip(modelNames, outputDirs):
  try:
    print("Started to Work...")
    print("Model Name:", modelName)
    print("Output Dir:", outputDir)
    PretrainedVisionTransformer(splitFolder, modelName, outputDir, **configs)
    VisionTransformerInference(splitFolder, outputDir, splitType="test")
  except Exception as e:
    print(e)
    continue
