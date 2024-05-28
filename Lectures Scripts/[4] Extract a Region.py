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
import matplotlib.pyplot as plt

svsFile = r"A02.svs"
xmlFile = r"A02.xml"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)
anList = []  # List of annotations.

# Read the XML file.
tree = ET.parse(xmlFile)
root = tree.getroot()
annotations = root.findall(".//Annotation")
for annotation in annotations:
  regions = annotation.findall(".//Region")
  for region in regions:
    vertices = region.findall(".//Vertex")
    coords = [
      (
        int(float(vertex.attrib["X"])),
        int(float(vertex.attrib["Y"]))
      )
      for vertex in vertices
    ]
    anList.append(
      {
        "Text"  : region.attrib["Text"],
        "Coords": coords
      }
    )

# Extract a region from the SVS file.
region = anList[0]
regionCoords = region["Coords"]
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

regionImage = slide.read_region(
  (min(regionX), min(regionY)),  # Top left corner.
  0,  # Level 0.
  (regionWidth, regionHeight),  # Width x Height.
)

# Extract the region from the mask.
regionImage = np.array(regionImage).astype(np.uint8)
regionImage = cv2.cvtColor(regionImage, cv2.COLOR_RGBA2RGB)
roi = cv2.bitwise_and(regionImage, regionImage, mask=regionMask)
roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

# Display the region.
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(regionMask, cmap="gray")
plt.axis("off")
plt.title("Region Mask")
plt.subplot(1, 3, 2)
plt.imshow(regionImage)
plt.axis("off")
plt.title("Region Image")
plt.subplot(1, 3, 3)
plt.imshow(roi)
plt.axis("off")
plt.title("Region ROI")
plt.show()
