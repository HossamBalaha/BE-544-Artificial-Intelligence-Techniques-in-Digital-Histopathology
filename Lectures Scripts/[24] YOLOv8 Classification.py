# https://stackoverflow.com/questions/75111196/yolov8-runtimeerror-an-attempt-has-been-made-to-start-a-new-process-before-th
# https://docs.ultralytics.com/tasks/classify/#models
# https://docs.ultralytics.com/usage/cfg/#export-settings

# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import splitfolders, os, cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import Heatmap

if __name__ == "__main__":
  datasetDirName = r"BACH Updated Extracted\ROIs_0_256_256_32_32"
  splitFolder = r"BACH Updated Extracted\ROIs_0_256_256_32_32_Split"
  baseDir = r"History"

  if (not os.path.exists(splitFolder)):
    splitfolders.ratio(
      os.path.join(baseDir, datasetDirName),
      output=os.path.join(baseDir, splitFolder),
      seed=1337,
      ratio=(0.8, 0.1, 0.1),
      group_prefix=None,
      move=False,
    )

  inputShape = (256, 256)
  epochs = 50

  for modelKeyword in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"][::-1]:
    model = YOLO(f"{modelKeyword}-cls.pt")
    model.train(
      data=splitFolder,
      epochs=epochs,
      imgsz=inputShape[0],
      plots=True,
      save=True,
      name=f"{modelKeyword}-cls-{inputShape[0]}",
    )

    classes = model.names

    # Validate the model.
    metrics = model.val(
      plots=False,
      save=False,
      name=f"{modelKeyword}-cls-{inputShape[0]}",
    )

    print("Top-1 Metrics:", metrics.top1)
    print("Top-5 Metrics:", metrics.top5)

    print("Exporting model...")
    model.save(os.path.join(baseDir, rf"runs\classify\{modelKeyword}-cls-{inputShape[0]}", f"model.keras"))

    testDir = os.path.join(baseDir, splitFolder, "test")
    rndCls = classes[np.random.randint(0, len(classes))]
    rndImg = np.random.choice(os.listdir(os.path.join(testDir, rndCls)))
    rndImgPath = os.path.join(testDir, rndCls, rndImg)
    print("Random Image:", rndImgPath)

    img = cv2.imread(rndImgPath)
    img = cv2.resize(img, inputShape, interpolation=cv2.INTER_CUBIC)

    print("Predicting...")
    pred = model.predict(
      [img],
      # visualize=True,
      imgsz=inputShape[0],
      save=False,
      classes=model.names,
      name=f"{modelKeyword}-cls-{inputShape[0]}",
    )
    print("Prediction:", pred)
