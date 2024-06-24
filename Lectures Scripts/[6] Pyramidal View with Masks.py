# Author: Hossam Magdy Balaha
# Date: May 30th, 2024
# Permissions and Citation: Refer to the README file.

# -------------------------------------------------- #
# https://openslide.org/api/python/#installing
OPENSLIDE_PATH = r"C:\openslide-bin-4.0.0.3-windows-x64\bin"

import os

if (hasattr(os, "add_dll_directory")):
  with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
else:
  import openslide
# -------------------------------------------------- #

import openslide, cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


def ExtractAnnotations(xmlFile, dFactor=1.0):
  anList = []  # List of annotations.

  # Read and parse the XML file.
  tree = ET.parse(xmlFile)
  root = tree.getroot()
  annotations = root.findall(".//Annotation")
  for annotation in annotations:
    regions = annotation.findall(".//Region")
    for region in regions:
      vertices = region.findall(".//Vertex")
      # Scale the coordinates by the downsample factor.
      coords = [
        (
          int(float(vertex.attrib["X"]) / dFactor),
          int(float(vertex.attrib["Y"]) / dFactor)
        )
        for vertex in vertices
      ]
      anList.append(
        {
          "Text"  : region.attrib["Text"],
          "Coords": coords
        }
      )

  return anList


def ExtractMask(regionCoords, padFactor=1.0):
  regionX = [x for x, y in regionCoords]
  regionY = [y for x, y in regionCoords]
  regionWidth = max(regionX) - min(regionX)
  regionHeight = max(regionY) - min(regionY)

  # Mask the region with a contour B/W.
  regionXShifted = [x - min(regionX) for x in regionX]
  regionYShifted = [y - min(regionY) for y in regionY]
  regionCoordsShifted = [(x, y) for x, y in zip(regionXShifted, regionYShifted)]
  regionCoordsShifted = np.array(regionCoordsShifted, np.int32)
  regionMask = np.zeros((regionHeight, regionWidth))
  regionMask = cv2.fillPoly(regionMask, [regionCoordsShifted], 255)
  regionMask = regionMask.astype(np.uint8)

  remainingWidth = regionWidth * padFactor - regionWidth
  remainingHeight = regionHeight * padFactor - regionHeight

  regionMask = cv2.copyMakeBorder(
    regionMask,
    remainingHeight // 2,
    remainingHeight - remainingHeight // 2,
    remainingWidth // 2,
    remainingWidth - remainingWidth // 2,
    cv2.BORDER_CONSTANT,
    value=0
  )

  regionXShifted = [x + remainingWidth // 2 for x in regionXShifted]
  regionYShifted = [y + remainingHeight // 2 for y in regionYShifted]

  return (
    regionMask, regionWidth, regionHeight,
    regionX, regionY, regionXShifted, regionYShifted
  )


svsFile = r"A02.svs"
xmlFile = r"A02.xml"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Find the properties of the slide.
slideLevels = slide.level_count

(width, height) = (512, 512)
annotationIndex = 0  # Index of the annotation.

plt.figure()
for i in range(slideLevels):
  # Calculate the downsample ratio with reference to the highest resolution.
  dRatio = int(slide.level_downsamples[i] / slide.level_downsamples[0])

  # Calculate the downsample factor.
  factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
  factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

  # Extract the annotations.
  anList = ExtractAnnotations(xmlFile, dFactor=dRatio)

  region = anList[annotationIndex]
  regionCoords = region["Coords"]
  (
    mask, regionWidth, regionHeight,
    regionX, regionY, regionXShifted, regionYShifted
  ) = ExtractMask(
    regionCoords,
    padFactor=dRatio,
  )

  # For the mask.
  xMask = np.min(regionXShifted)
  yMask = np.min(regionYShifted)
  xNewMask = xMask - factorWidth
  yNewMask = yMask - factorHeight
  regionMask = mask[yNewMask:yNewMask + height, xNewMask:xNewMask + width]

  # For the region.
  # Changed x to np.min(regionX) * dRatio
  # This to be with reference to the mask.
  xRegion = dRatio * (np.min(regionX))
  yRegion = dRatio * (np.min(regionY))
  xNewRegion = xRegion - dRatio * factorWidth
  yNewRegion = yRegion - dRatio * factorHeight

  regionSlide = slide.read_region(
    (xNewRegion, yNewRegion),
    i,
    (width, height),
  )
  regionSlide = regionSlide.convert("RGB")
  regionSlide = np.array(regionSlide)

  roi = cv2.bitwise_and(regionSlide, regionSlide, mask=regionMask)
  roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

  plt.subplot(3, slideLevels, i + 1)
  plt.imshow(regionSlide)
  plt.axis("off")
  plt.tight_layout()
  plt.title(f"Level {i} ({dRatio}x)")

  plt.subplot(3, slideLevels, 3 + i + 1)
  plt.imshow(regionMask, cmap="gray")
  plt.axis("off")
  plt.tight_layout()
  plt.title(f"Level {i} Mask")

  plt.subplot(3, slideLevels, 6 + i + 1)
  plt.imshow(roi, cmap="gray")
  plt.axis("off")
  plt.tight_layout()
  plt.title(f"Level {i} ROI")

plt.show()
