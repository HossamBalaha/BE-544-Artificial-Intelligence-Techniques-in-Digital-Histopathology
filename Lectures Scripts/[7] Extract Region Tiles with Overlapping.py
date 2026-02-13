'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

import openslide, os, cv2
# Import helper functions for parsing and mapping annotations from the helpers module.
from HMB_Spring_2026_Helpers import (
  ExtractBACHAnnotationsFromXML,
  MapAnnotationsFromLevelAToLevelB,
  ExtractWSIROI,
)

svsFile = r"Data/A02.svs"
xmlFile = r"Data/A02.xml"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Find the properties of the slide.
levels = slide.level_count

(width, height) = (512, 512)
annotationIndex = 0  # Index of the annotation.
overlapWidth = 256  # Width of the overlap.
overlapHeight = 256  # Height of the overlap.
threshold = 0.7  # Threshold of the mask.

storagePath = "Output"
os.makedirs(storagePath, exist_ok=True)

for level in range(levels):
  print(f"Processing Level {level} with downsample ratio {slide.level_downsamples[level]}...")

  # Calculate the downsample ratio with reference to the highest resolution.
  dRatio = int(slide.level_downsamples[level] / slide.level_downsamples[0])

  # Extract BACH annotations from the XML file at the base level.
  anList = ExtractBACHAnnotationsFromXML(xmlFile, verbose=False)

  # Select the annotation dictionary using the specified index.
  region = anList[annotationIndex]
  # Map the annotation coordinates from level A to the current level B using helper function.
  mappedRegion = MapAnnotationsFromLevelAToLevelB(region, dRatio)

  # Extract the x coordinates from the polygon vertices.
  regionX = [x for x, y in region["Coords"]]
  # Extract the y coordinates from the polygon vertices.
  regionY = [y for x, y in region["Coords"]]
  # Compute the width of the bounding box of the region.
  regionWidth = max(regionX) - min(regionX)
  # Compute the height of the bounding box of the region.
  regionHeight = max(regionY) - min(regionY)

  for x in range(0, regionWidth * dRatio, width - overlapWidth):
    for y in range(0, regionHeight * dRatio, height - overlapHeight):
      (regionSlide, regionMask, roi) = ExtractWSIROI(
        # An OpenSlide object representing the whole-slide image (WSI) or an object with a `read_region` method and `dimensions` attribute.
        slide,
        # A dictionary with a "Coords" key containing a list of (x, y) tuples representing the vertices of the polygon.
        region=mappedRegion,
        level=level,  # The pyramid level to read the region from (default is 0 for highest resolution).
        dRatio=dRatio,  # The downsample ratio to apply to the coordinates (default is 1.0 for no change).
        width=width,  # The width of the output region to extract in pixels (default is 512).
        height=height,  # The height of the output region to extract in pixels (default is 512).
        startX=x,  # The x-coordinate offset to apply to the region when reading from the slide (default is 0).
        startY=x,  # The y-coordinate offset to apply to the region when reading from the slide (default is 0).
        # The minimum percentage of the mask that must be present in the extracted region to consider it valid (default is 0.0 for no threshold).
        maskThreshold=threshold,
      )

      # if (regionSlide is None or regionMask is None or roi is None):
      #   print(f"Skipping region at (x={x}, y={y}) due to insufficient mask coverage.")
      #   continue

      category = region['Text']
      postfix = f"{level}_{width}_{height}_{overlapWidth}_{overlapHeight}_{category}"
      fileName = f"{svsFile.split('.')[0]}_{annotationIndex}_{x}_{y}.jpg"

      roisPath = f"{storagePath}/ROIs_{postfix}"
      os.makedirs(roisPath, exist_ok=True)
      cv2.imwrite(os.path.join(roisPath, fileName), roi)

      tilesPath = f"{storagePath}/Tiles_{postfix}"
      os.makedirs(tilesPath, exist_ok=True)
      cv2.imwrite(os.path.join(tilesPath, fileName), regionSlide)

      masksPath = f"{storagePath}/Masks_{postfix}"
      os.makedirs(masksPath, exist_ok=True)
      cv2.imwrite(os.path.join(masksPath, fileName), regionMask)

print("Processing completed for all levels and regions.")
