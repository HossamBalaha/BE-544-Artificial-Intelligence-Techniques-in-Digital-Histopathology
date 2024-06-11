# Author: Hossam Magdy Balaha
# Date: May 30th, 2024

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


def ExtractRegionTiles(
  region, factorWidth, factorHeight, dRatio,
  width, height, overlapWidth, overlapHeight,
  level, slide, svsFile, annotationIndex,
  tolerance=0.8,
):
  regionCoords = region["Coords"]

  (
    mask, regionWidth, regionHeight,
    regionX, regionY, regionXShifted, regionYShifted
  ) = ExtractMask(
    regionCoords,
    padFactor=dRatio,
  )

  for x in range(0, regionWidth * dRatio, width - overlapWidth):
    for y in range(0, regionHeight * dRatio, height - overlapHeight):
      # For the mask.
      xMask = np.min(regionXShifted) + x
      yMask = np.min(regionYShifted) + y
      xNewMask = xMask - factorWidth
      yNewMask = yMask - factorHeight
      regionMask = mask[yNewMask:yNewMask + height, xNewMask:xNewMask + width]

      if (
        (regionMask.shape[0] != height) or
        (regionMask.shape[1] != width)
      ):
        continue

      # Check the percentage of the mask.
      if (np.sum(regionMask) < (tolerance / dRatio ** 2) * regionMask.size * 255):
        continue

      # For the region.
      # Changed x to np.min(regionX) * dRatio
      # This to be with reference to the mask.
      xRegion = dRatio * (np.min(regionX) + x)
      yRegion = dRatio * (np.min(regionY) + y)
      xNewRegion = xRegion - dRatio * factorWidth
      yNewRegion = yRegion - dRatio * factorHeight

      regionSlide = slide.read_region(
        (xNewRegion, yNewRegion),
        level,
        (width, height),
      )
      regionSlide = regionSlide.convert("RGB")
      regionSlide = np.array(regionSlide)

      roi = cv2.bitwise_and(regionSlide, regionSlide, mask=regionMask)
      roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

      dirName = f"Output/ROIs_{level}_{width}_{height}_{overlapWidth}_{overlapHeight}_{region['Text']}"
      fileName = f"{svsFile.split('.')[0]}_{annotationIndex}_{x}_{y}.jpg"
      if (not os.path.exists(dirName)):
        os.makedirs(dirName)
      cv2.imwrite(os.path.join(dirName, fileName), roi)

      dirName = f"Output/Tiles_{level}_{width}_{height}_{overlapWidth}_{overlapHeight}_{region['Text']}"
      fileName = f"{svsFile.split('.')[0]}_{annotationIndex}_{x}_{y}.jpg"
      if (not os.path.exists(dirName)):
        os.makedirs(dirName)
      cv2.imwrite(os.path.join(dirName, fileName), regionSlide)

      dirName = f"Output/Masks_{level}_{width}_{height}_{overlapWidth}_{overlapHeight}_{region['Text']}"
      fileName = f"{svsFile.split('.')[0]}_{annotationIndex}_{x}_{y}.jpg"
      if (not os.path.exists(dirName)):
        os.makedirs(dirName)
      cv2.imwrite(os.path.join(dirName, fileName), regionMask)


svsFile = r"A02.svs"
xmlFile = r"A02.xml"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Find the properties of the slide.
levels = slide.level_count

(width, height) = (512, 512)
overlapWidth = 256  # Width of the overlap.
overlapHeight = 256  # Height of the overlap.

for level in range(levels):
  # Calculate the downsample ratio with reference to the highest resolution.
  dRatio = int(slide.level_downsamples[level] / slide.level_downsamples[0])

  # Calculate the downsample factor.
  factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
  factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

  # Extract the annotations.
  anList = ExtractAnnotations(xmlFile, dFactor=dRatio)

  for annotationIndex in range(len(anList)):  # len(anList)
    region = anList[annotationIndex]

    # Extract the region tiles.
    ExtractRegionTiles(
      region, factorWidth, factorHeight, dRatio,
      width, height, overlapWidth, overlapHeight,
      level, slide, svsFile, annotationIndex,
      tolerance=0.7,
    )
