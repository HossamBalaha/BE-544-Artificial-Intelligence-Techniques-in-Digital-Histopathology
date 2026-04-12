'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# # -------------------------------------------------- #
# # https://openslide.org/download/
# # https://openslide.org/api/python/#installing
# # Make sure to set the correct path to the OpenSlide binaries.
# # This is required on Windows systems to load the OpenSlide DLLs.
# OPENSLIDE_PATH = r"C:\openslide-bin-4.0.0.11-windows-x64\bin"
#
# import os
#
# if (hasattr(os, "add_dll_directory")):
#   with os.add_dll_directory(OPENSLIDE_PATH):
#     import openslide
# else:
#   import openslide
# # -------------------------------------------------- #

# Import necessary libraries.
import os, cv2, openslide, tqdm, patchify, torch
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import *
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
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


def ExtractBACHAnnotationsFromXML(xmlFile, verbose=True):
  r'''
  Extract annotations from a BACH XML file.
  
  Parameters:
    xmlFile (str): Path to the XML file containing annotations.
    verbose (bool): Whether to print debug information.
    
  Returns:
    list: A list of dictionaries, each containing "Text" and "Coords" keys.
  '''

  if (not os.path.exists(xmlFile)):
    raise FileNotFoundError(f"XML file not found: {xmlFile}")

  # Parse the XML file into an ElementTree object.
  tree = ET.parse(xmlFile)
  # Get the root element of the parsed XML tree.
  root = tree.getroot()
  # Initialize the list that will hold parsed annotations.
  anList = []  # List of annotations.

  # Find all Annotation elements anywhere in the XML tree.
  annotations = root.findall(".//Annotation")
  if (verbose):
    # Print the number of top-level Annotation elements found.
    print("Number of annotations: ", len(annotations))

  # If no annotations were found, return the empty list.
  if (len(annotations) == 0):
    return anList

  # Iterate over each Annotation element found in the XML.
  for annotation in annotations:
    # Find all Region elements inside the current Annotation element.
    regions = annotation.findall(".//Region")
    if (verbose):
      # Print how many Region elements were found in this Annotation.
      print("- Number of regions: ", len(regions))
    # Iterate over each Region element inside the current Annotation.
    for region in regions:
      if (verbose):
        # Print the textual label associated with the current Region.
        print("-- Region Text: ", region.attrib["Text"])
      # Find all Vertex elements that define the polygon of the current Region.
      vertices = region.findall(".//Vertex")
      if (verbose):
        # Print how many Vertex elements define this Region.
        print("-- Number of vertices: ", len(vertices))

      # Build a list of integer (x, y) coordinate tuples from the Vertex attributes.
      coords = [
        (
          int(float(vertex.attrib["X"])),
          int(float(vertex.attrib["Y"]))
        )
        for vertex in vertices
      ]

      # Append a dictionary with the Region text and coordinates to the annotations list.
      anList.append(
        {
          "Text"  : region.attrib["Text"],
          "Coords": coords
        }
      )

  return anList


def ExtractWSIRegion(slide, region):
  r'''
  Extract a region of interest (ROI) from a whole-slide image (WSI) using OpenSlide.
  The extracted region is from the highest resolution level (level 0) and is masked
  according to the polygon defined by the annotation.

  Parameters:
    slide (openslide.OpenSlide or slide-like): The OpenSlide object representing the WSI or an object with a `read_region` method and `dimensions` attribute.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples.

  Returns:
    tuple: A tuple containing:
      - regionImage (numpy.ndarray): The extracted region image as a NumPy RGB array (H,W,3) uint8.
      - regionMask (numpy.ndarray): The binary mask for the region as a NumPy uint8 array (H,W) with 0/255 values.
      - roi (numpy.ndarray): The extracted region of interest (ROI) as a NumPy RGB array (H,W,3) uint8.
  '''

  # Validate the input region dictionary.
  if (region is None):
    raise ValueError("Region cannot be None.")
  if ("Coords" not in region):
    raise KeyError("Region dictionary must contain `Coords` key.")

  # Validate that the slide is an OpenSlide object.
  if (not isinstance(slide, openslide.OpenSlide)):
    raise TypeError("Slide must be an OpenSlide object.")
  if (getattr(slide, "closed", False)):
    raise ValueError("Slide is closed.")

  # Extract the list of coordinate tuples for the selected region.
  regionCoords = region["Coords"]
  # Ensure there are enough points to form a polygon.
  if (not regionCoords) or (len(regionCoords) < 1):
    raise ValueError("Region `Coords` must contain at least one (x,y) point.")

  # Build lists of x and y coordinates from the region coordinates.
  regionX = [int(x) for x, y in regionCoords]
  regionY = [int(y) for x, y in regionCoords]

  # Compute inclusive bounding box for the region in pixels (add +1 to include boundary pixels).
  minX = min(regionX)
  maxX = max(regionX)
  minY = min(regionY)
  maxY = max(regionY)
  regionWidth = maxX - minX + 1
  regionHeight = maxY - minY + 1

  # Reject degenerate boxes as they cannot form valid masks or images.
  if ((regionWidth <= 0) or (regionHeight <= 0)):
    raise ValueError("Computed region width/height must be positive.")

  # Shift the region coordinates so the polygon starts at (0,0) for mask creation.
  regionXShifted = [x - minX for x in regionX]
  regionYShifted = [y - minY for y in regionY]

  # Combine shifted x and y lists into a list of (x,y) tuples for the polygon.
  regionCoordsShifted = [(x, y) for x, y in zip(regionXShifted, regionYShifted)]

  # Convert the polygon coordinate list to a NumPy array of type int32 for OpenCV.
  regionCoordsShifted = np.array(regionCoordsShifted, np.int32)

  # Create an empty mask array of zeros with the region bounding box shape and uint8 dtype.
  regionMask = np.zeros((regionHeight, regionWidth), dtype=np.uint8)

  # Fill the polygon area on the mask with 255 to create a binary mask.
  cv2.fillPoly(regionMask, [regionCoordsShifted], 255)

  # Read the region image from the slide at level 0 using the bounding box top-left corner and size.
  regionImage = slide.read_region(
    (minX, minY),  # Top left corner.
    0,
    (regionWidth, regionHeight),  # Width x Height.
  )

  # Convert the PIL.Image returned by read_region to a NumPy uint8 array.
  regionImage = np.array(regionImage).astype(np.uint8)
  # Convert the image from RGBA to RGB color space for display.
  regionImage = cv2.cvtColor(regionImage, cv2.COLOR_RGBA2RGB)
  # Apply the mask to the region image to isolate the ROI using bitwise_and.
  roi = cv2.bitwise_and(regionImage, regionImage, mask=regionMask)
  # Convert the ROI from RGBA to RGB in case the image still contains alpha.
  roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

  # Return a tuple containing the region image, mask, and extracted ROI.
  return regionImage, regionMask, roi


def ExtractPyramidalWSITiles(
  slide,
  x=0,
  y=0,
  width=512,
  height=512,
):
  # Get the number of pyramid levels in the slide.
  slideLevels = slide.level_count

  # Create a new matplotlib figure to plot the regions from each level.
  plt.figure()

  # A dictionary to hold the extracted tiles for each level, keyed by level index.
  tiles = {}

  # Iterate over each level in the slide pyramid.
  for slideLevel in range(slideLevels):
    # Calculate the downsample ratio relative to the highest resolution level.
    dRatio = int(slide.level_downsamples[slideLevel] / slide.level_downsamples[0])

    # Calculate the horizontal offset factor used to center the crop at lower resolutions.
    factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
    # Calculate the vertical offset factor used to center the crop at lower resolutions.
    factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

    # Compute the new x-coordinate for the region at the current level.
    xNew = x - dRatio * factorWidth
    # Compute the new y-coordinate for the region at the current level.
    yNew = y - dRatio * factorHeight

    # print(f"Level {i}: Downsample Ratio={dRatio}, xNew={xNew}, yNew={yNew}")

    # Read a region from the slide at the given level and coordinates.
    regionSlide = slide.read_region(
      (xNew, yNew),
      slideLevel,
      (width, height),
    )
    # Convert the returned PIL image to RGB mode.
    regionSlide = regionSlide.convert("RGB")
    # Convert the PIL image to a NumPy array for manipulation and display.
    regionSlide = np.array(regionSlide)

    # Select the subplot for displaying the full region at this pyramid level.
    plt.subplot(2, slideLevels, slideLevel + 1)
    # Render the region image in the subplot.
    plt.imshow(regionSlide)
    # Disable axis ticks and labels for the image subplot.
    plt.axis("off")
    # Adjust subplot layout to minimize overlaps.
    plt.tight_layout()
    # Set a title for the subplot including level and downsample factor.
    plt.title(f"Level {slideLevel} ({dRatio}x)")

    # Crop the rendered region to verify the corresponding area at the current level.
    verify = regionSlide[
      factorHeight:factorHeight + height // dRatio,
      factorWidth:factorWidth + width // dRatio,
    ]

    # Select the subplot for displaying the verification crop.
    plt.subplot(2, slideLevels, 3 + slideLevel + 1)
    # Render the verification crop in the subplot.
    plt.imshow(verify)
    # Disable axis ticks and labels for the verification subplot.
    plt.axis("off")
    # Adjust layout for the verification subplot.
    plt.tight_layout()
    # Set a title for the verification subplot indicating the level verified.
    plt.title(f"Cropped to Verify Level {slideLevel}")

    # Store the extracted tile for the current level in the tiles dictionary.
    tiles[slideLevel] = regionSlide

  # Get the current figure to return it for display or saving.
  figToReturn = plt.gcf()

  return tiles, figToReturn


def PrepareAnnotationsForLevel(annotation, dFactor=1.0):
  r'''
  Map annotation coordinates from one pyramid level to another by applying a downsample factor.

  Parameters:
    annotation (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples.
    dFactor (float): The downsample factor to apply to the coordinates (default is 1.0 for no change).

  Returns:
    dict: A new annotation dictionary with the same "Text" and scaled "Coords".
  '''

  if (annotation is None):
    raise ValueError("Annotation cannot be None.")
  if ("Coords" not in annotation):
    raise KeyError("Annotation dictionary must contain `Coords` key.")

  # Extract the original coordinates from the annotation.
  originalCoords = annotation["Coords"]

  # Scale the coordinates by the downsample factor.
  mappedCoords = [
    (
      int(float(a) / dFactor),
      int(float(b) / dFactor)
    )
    for (a, b) in originalCoords
  ]

  # Shift the mapped coordinates so the polygon starts at (0,0) for mask creation.
  minX = min(x for x, y in mappedCoords)
  minY = min(y for x, y in mappedCoords)
  maxX = max(x for x, y in mappedCoords)
  maxY = max(y for x, y in mappedCoords)
  shiftedCoords = [
    (x - minX, y - minY)
    for (x, y) in mappedCoords
  ]

  mask = np.zeros((maxY - minY, maxX - minX))
  mask = cv2.fillPoly(mask, [np.array(shiftedCoords, np.int32)], 255)

  # Pad the mask to match the size of the base mask.
  baseWidth = int((maxX - minX) * dFactor)
  baseHeight = int((maxY - minY) * dFactor)
  padX = (baseWidth - mask.shape[1]) // 2
  padY = (baseHeight - mask.shape[0]) // 2
  mask = cv2.copyMakeBorder(
    mask,
    top=padY,
    bottom=padY,
    left=padX,
    right=padX,
    borderType=cv2.BORDER_CONSTANT,
    value=0
  )

  # Update the shifted coordinates to account for the padding added to the mask.
  shiftedCoords = [
    (x + padX, y + padY)
    for (x, y) in shiftedCoords
  ]

  # Return a new annotation dictionary with the same text and mapped coordinates.
  return {
    "Text"         : annotation.get("Text", ""),
    "Coords"       : mappedCoords,
    "MinX"         : minX,
    "MinY"         : minY,
    "MaxX"         : maxX,
    "MaxY"         : maxY,
    "Width"        : maxX - minX,
    "Height"       : maxY - minY,
    "ShiftedCoords": shiftedCoords,
    "dFactor"      : dFactor,
    "Mask"         : mask.astype(np.uint8),
  }


def ExtractRegionTiles(
  slide,
  region,
  width=512,
  height=512,
  overlapWidth=0,
  overlapHeight=0,
  storageDir=None,
  maxTiles=None,
  addPlots=True,
  prefix="",
  blackRatioThreshold=0.90,
  removeBackgroundTiles=True,
  convertBlackToWhite=True,
):
  r'''
  Extract tiles from a specified region of a whole-slide image (WSI) across all pyramid levels,
  applying annotation masks and saving results. The function handles the mapping of annotations
  to each level, extracts tiles, applies masks, and optionally saves the tiles, masks, and ROIs to disk.

  Parameters:
    slide (openslide.OpenSlide): The OpenSlide object representing the WSI.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples representing the annotation polygon.
    width (int): The width of the tiles to extract in pixels (default is 512).
    height (int): The height of the tiles to extract in pixels (default is 512).
    overlapWidth (int): The horizontal overlap between tiles in pixels (default is 0).
    overlapHeight (int): The vertical overlap between tiles in pixels (default is 0).
    storageDir (str or None): The directory path to save the extracted tiles, masks, and ROIs. If None, no files will be saved (default is None).
    maxTiles (int or None): The maximum number of tiles to extract for the region. If None, all tiles will be extracted (default is None).
    addPlots (bool): Whether to create and save plots visualizing the tiles, masks, and ROIs (default is True).
    prefix (str): A string prefix to add to saved file names for organization (default is an empty string).
    blackRatioThreshold (float): The maximum allowed ratio of black pixels in a tile to be considered valid (default is 0.90). Tiles with a higher ratio will be skipped.
    removeBackgroundTiles (bool): Whether to skip tiles that are considered background based on the black pixel ratio (default is True).
    convertBlackToWhite (bool): Whether to convert black pixels to white in the ROI before background analysis to avoid skewing metrics (default is True).
  '''

  # Create output directories when a storage directory is provided.
  if (storageDir is not None):
    if (addPlots):
      # Compose the plots directory path and ensure it exists.
      plotsDir = os.path.join(storageDir, "Plots")
      os.makedirs(plotsDir, exist_ok=True)
    else:
      plotsDir = None
    # Compose the tiles directory path and ensure it exists.
    tilesDir = os.path.join(storageDir, "Tiles")
    os.makedirs(tilesDir, exist_ok=True)
    # Compose the masks directory path and ensure it exists.
    masksDir = os.path.join(storageDir, "Masks")
    os.makedirs(masksDir, exist_ok=True)
    # Compose the ROIs directory path and ensure it exists.
    roisDir = os.path.join(storageDir, "ROIs")
    os.makedirs(roisDir, exist_ok=True)
    # Pre-create subdirectories for each pyramid level for tiles, masks, and ROIs.
    for level in range(slide.level_count):
      os.makedirs(os.path.join(tilesDir, f"Level_{level}"), exist_ok=True)
      os.makedirs(os.path.join(masksDir, f"Level_{level}"), exist_ok=True)
      os.makedirs(os.path.join(roisDir, f"Level_{level}"), exist_ok=True)
  else:
    plotsDir = tilesDir = masksDir = roisDir = None

  # Initialize a dictionary to hold mapping data for all levels.
  mappingData = {}
  # Build mapping data for each pyramid level by preparing annotations for that level.
  for level in range(slide.level_count):
    # Compute the integer downsample factor relative to level 0.
    dFactor = int(slide.level_downsamples[level] / slide.level_downsamples[0])
    # Prepare the annotation scaled/mapped for the current level.
    annotation = PrepareAnnotationsForLevel(region, dFactor)
    # Store the prepared annotation into the mapping dictionary keyed by the level.
    mappingData[level] = annotation

  # Extract the start coordinates and dimensions of the region at the base level.
  regionStartX = mappingData[0]["MinX"]
  regionStartY = mappingData[0]["MinY"]
  regionWidth = mappingData[0]["Width"]
  regionHeight = mappingData[0]["Height"]
  category = mappingData[0]["Text"]

  xProgressBar = tqdm.tqdm(
    range(regionStartX, regionStartX + regionWidth, width - overlapWidth),
    desc="Processing X-axis",
    position=0,
  )
  yProgressBar = tqdm.tqdm(
    range(regionStartY, regionStartY + regionHeight, height - overlapHeight),
    desc="Processing Y-axis",
    leave=False,
    position=1,
  )
  # Initialize a counter to keep track of the number of tiles processed (optional, can be used for maxTiles limit).
  counter = 0
  for x in xProgressBar:
    for y in yProgressBar:
      startX = x - regionStartX
      startY = y - regionStartY

      # Extract pyramidal tiles for the current window and receive the plotting figure.
      tiles, fig1 = ExtractPyramidalWSITiles(
        slide,
        x=x,
        y=y,
        width=width,
        height=height,
      )
      # Close the temporary figure to free-associated resources.
      plt.close(fig1)
      plt.gcf().clear()  # Clear the current figure to reset the plotting state for the next iteration.

      # Create a shapely polygon for the base-level annotation to test intersection with the tile.
      baseCoordsPolygon = Polygon(mappingData[0]["ShiftedCoords"])
      # Define the polygon for the current tile region in the region-local coordinate space.
      tileRegion = Polygon([
        (startX, startY),
        (startX + width, startY),
        (startX + width, startY + height),
        (startX, startY + height),
      ])
      # Skip this tile if it does not intersect with the annotation polygon.
      if (not baseCoordsPolygon.intersects(tileRegion)):
        # print(f"Tile at x: {x}, y: {y} does not intersect with annotation region. Skipping.")
        continue

      # Prepare a plotting figure if storage is enabled so we can visualize results.
      if (addPlots and storageDir is not None):
        plt.figure(figsize=(12, 3 * slide.level_count))

      whatToStore = {}

      # Iterate over each pyramid level to crop masks and produce ROIs for saving/plotting.
      for level in range(slide.level_count):
        # Retrieve the tile image for the current level from the extracted tiles.
        levelTile = tiles[level]
        # Retrieve the precomputed mask for the current level from mappingData.
        levelMask = mappingData[level]["Mask"]
        # Retrieve the shifted coordinates used to align the mask for cropping.
        levelShiftedCoords = mappingData[level]["ShiftedCoords"]
        # Compute the minimum x coordinate of the shifted coordinates for the mask alignment.
        levelStartX = min(coord[0] for coord in levelShiftedCoords)
        # Compute the minimum y coordinate of the shifted coordinates for the mask alignment.
        levelStartY = min(coord[1] for coord in levelShiftedCoords)

        # Compute the downsample ratio integer for the current level relative to level 0.
        dRatio = int(slide.level_downsamples[level] / slide.level_downsamples[0])

        # Calculate the width padding factor to center crops at lower resolutions.
        factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
        # Calculate the height padding factor to center crops at lower resolutions.
        factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

        # Compute the x coordinate in the level mask coordinate space for cropping.
        levelX = levelStartX - factorWidth + (startX // dRatio)
        # Compute the y coordinate in the level mask coordinate space for cropping.
        levelY = levelStartY - factorHeight + (startY // dRatio)
        # Crop the mask tile from the full level mask using the computed coordinates and the requested size.
        levelMaskTile = levelMask[levelY:levelY + height, levelX:levelX + width]
        # Compute padding values needed to center the mask tile inside the level tile if sizes differ.
        padX = (levelTile.shape[1] - levelMaskTile.shape[1]) // 2
        padY = (levelTile.shape[0] - levelMaskTile.shape[0]) // 2
        # Pad the mask tile so it matches the tile image size using a constant zero border.
        levelMaskTile = cv2.copyMakeBorder(
          levelMaskTile,  # Input mask tile to be padded.
          top=padY,  # Number of pixels to pad on the top of the mask tile.
          bottom=padY,  # Number of pixels to pad on the bottom of the mask tile.
          left=padX,  # Number of pixels to pad on the left of the mask tile.
          right=padX,  # Number of pixels to pad on the right of the mask tile.
          borderType=cv2.BORDER_CONSTANT,  # Type of border to use for padding (constant value).
          value=0,  # The constant value to use for padding (0 for black).
        )
        # Ensure the padded mask tile is of type uint8 for proper masking operations.
        levelMaskTile = levelMaskTile.astype(np.uint8)

        if ((levelMaskTile.shape[0] != levelTile.shape[0]) or (levelMaskTile.shape[1] != levelTile.shape[1])):
          # Close the created figure.
          plt.close()
          whatToStore = {}
          break

        blackRatio = np.sum(levelMaskTile == 0) / levelMaskTile.size
        if ((level == 0) and (blackRatio > blackRatioThreshold)):
          # Close the created figure.
          plt.close()
          whatToStore = {}
          break

        # Compute the masked ROI by applying the binary mask to the tile image using a bitwise AND.
        levelROI = cv2.bitwise_and(levelTile, levelTile, mask=levelMaskTile)

        if (convertBlackToWhite):
          # Convert black pixels to white in the ROI.
          levelROI[levelROI == 0] = 255

        if ((level == 0) and (removeBackgroundTiles)):
          isBackground, metrics = IsBackgroundTile(
            None,
            image=levelROI.copy(),
            entropyThreshold=5.5,
            colorVarianceThreshold=1500,
            tissueAreaThreshold=0.20,
            convertBlackToWhite=convertBlackToWhite,
          )
          if (isBackground):
            # Close the created figure.
            plt.close()
            whatToStore = {}
            break

        whatToStore[level] = {
          "Tile": levelTile,
          "Mask": levelMaskTile,
          "ROI" : levelROI,
        }

      # Check if we have valid data to store for all levels before attempting to save or plot.
      if (whatToStore):
        for level in range(slide.level_count):
          levelTile = whatToStore[level]["Tile"]
          levelMaskTile = whatToStore[level]["Mask"]
          levelROI = whatToStore[level]["ROI"]

          # Save tile, mask, and ROI images to disk when storage is enabled.
          if (storageDir is not None):
            imgName = f"{level}_{x}_{y}_{width}x{height}_{overlapWidth}x{overlapHeight}"
            if (prefix):
              imgName = f"{prefix}_{imgName}"
            os.makedirs(os.path.join(tilesDir, f"Level_{level}", category), exist_ok=True)
            os.makedirs(os.path.join(masksDir, f"Level_{level}", category), exist_ok=True)
            os.makedirs(os.path.join(roisDir, f"Level_{level}", category), exist_ok=True)
            cv2.imwrite(os.path.join(tilesDir, f"Level_{level}", category, f"{imgName}.jpg"), levelTile)
            cv2.imwrite(os.path.join(masksDir, f"Level_{level}", category, f"{imgName}.jpg"), levelMaskTile)
            cv2.imwrite(os.path.join(roisDir, f"Level_{level}", category, f"{imgName}.jpg"), levelROI)

          # When storage is enabled, plot the tile, mask, overlay, and ROI for visual inspection.
          if (addPlots and storageDir is not None):
            plt.subplot(slide.level_count, 4, 1 + level * 4)
            plt.imshow(levelTile)
            plt.title("Tile")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 2 + level * 4)
            plt.imshow(levelMaskTile, cmap="gray")
            plt.title("Mask Tile")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 3 + level * 4)
            plt.imshow(levelTile)
            plt.imshow(levelMaskTile, alpha=0.5, cmap="jet")
            plt.title("Tile with Annotation Overlay")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 4 + level * 4)
            plt.imshow(levelROI)
            plt.title("ROI (Masked Tile)")
            plt.axis("off")
      else:
        # print(f"Tile at x: {x}, y: {y} has invalid mask or ROI. Skipping storage and plotting.")
        continue

      # When storage is enabled, finalize and save the plotted figure for the current tile.
      if (addPlots and storageDir is not None):
        imgName = f"{x}_{y}_{width}x{height}_{overlapWidth}x{overlapHeight}"
        if (prefix):
          imgName = f"{prefix}_{imgName}"
        os.makedirs(os.path.join(plotsDir, category), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plotsDir, category, f"{imgName}.png"), dpi=300, bbox_inches="tight")
        plt.close("all")
        plt.gcf().clear()

      counter += 1
      if ((maxTiles is not None) and (counter >= maxTiles)):
        print(f"Reached maximum tile limit of {maxTiles}. Stopping extraction.")
        return


def IsBackgroundTile(
  imagePath,  # Path to the tile image to analyze for background detection.
  image=None,  # Optional pre-loaded image as a NumPy array (H,W,3) uint8. If provided, imagePath will be ignored.
  # Threshold for Shannon entropy to detect uniformity. Adjust based on the expected variability in tissue tiles.
  entropyThreshold=5.5,
  # Threshold for color variance to detect lack of color diversity. Adjust based on the expected variability in tissue tiles.
  colorVarianceThreshold=1500,
  tissueAreaThreshold=0.20,  # Minimum ratio of tissue area to total area to consider the tile as non-background.
  convertBlackToWhite=True,  # Convert black pixels to white before analysis to avoid skewing the metrics.
):
  '''
  Detect background tiles using multiple criteria suitable for non-black backgrounds.

  Parameters:
    imagePath: Path to the tile image.
    image: Optional pre-loaded image as a NumPy array (H,W,3) uint8. If provided, imagePath will be ignored.
    entropyThreshold: Threshold for Shannon entropy to detect uniformity.
    colorVarianceThreshold: Threshold for color variance to detect lack of color diversity.
    tissueAreaThreshold: Threshold for the ratio of tissue area to total area.
    convertBlackToWhite: Whether to convert black pixels to white before analysis (default is True).

  Returns:
    bool: True if the tile is considered background, False otherwise.
    dict: A dictionary containing the computed metrics for debugging and analysis.
  '''

  import cv2
  import numpy as np
  from skimage.filters import threshold_otsu
  from skimage.measure import shannon_entropy

  if (image is None):
    if (not os.path.exists(imagePath)):
      raise FileNotFoundError(f"Image file not found: {imagePath}")
    image = cv2.imread(imagePath)

  # Convert black pixels to white to handle non-black backgrounds (e.g., white background in H&E slides).
  if (image is None):
    raise ValueError(f"Failed to load image from path: {imagePath}")

  if (convertBlackToWhite):
    image[image == 0] = 255

  # Convert to different color spaces for analysis.
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # 1. ENTROPY ANALYSIS (detects uniformity).
  entropyValue = shannon_entropy(gray)

  # 2. COLOR VARIANCE (H&E has characteristic pink/purple colors).
  colorVariance = np.var(image)

  # 3. TISSUE DETECTION using Otsu thresholding.
  # Invert since tissue is typically darker than background.
  thresh = threshold_otsu(gray)
  binary = gray < thresh
  tissueRatio = np.sum(binary) / binary.size

  # 4. SATURATION CHECK (H&E stained tissue has color saturation).
  saturation = hsv[:, :, 1]
  meanSaturation = np.mean(saturation)

  # 5. TEXTURE ANALYSIS (Laplacian variance).
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  textureVariance = np.var(laplacian)

  # Decision logic - tile is background if MOST criteria indicate background.
  backgroundScore = 0

  if (entropyValue < entropyThreshold):
    backgroundScore += 1
  if (colorVariance < colorVarianceThreshold):
    backgroundScore += 1
  if (tissueRatio < tissueAreaThreshold):
    backgroundScore += 1
  if (meanSaturation < 20):  # Low saturation = grayscale/white background.
    backgroundScore += 1
  if (textureVariance < 100):  # Low texture = smooth background.
    backgroundScore += 1

  # Consider background if 3 or more criteria agree.
  isBackground = backgroundScore >= 3

  return isBackground, {
    "entropy"        : entropyValue,
    "colorVariance"  : colorVariance,
    "tissueRatio"    : tissueRatio,
    "meanSaturation" : meanSaturation,
    "textureVariance": textureVariance,
    "backgroundScore": backgroundScore,
  }


def CalculateAllMetrics(cm):
  r'''
  Calculate a variety of classification metrics from a confusion matrix.

  This function computes per-class true positives (TP), false positives (FP),
  false negatives (FN) and true negatives (TN) from the provided confusion
  matrix and returns a dictionary containing macro-, micro- and
  class-weighted-averaged precision, recall, F1, accuracy and specificity.

  Parameters:
    cm (numpy.ndarray): Confusion matrix of shape (n_classes, n_classes).
      Rows correspond to ground-truth classes and columns to predicted
      classes. Each element cm[i, j] is the count of samples whose true
      label is i and predicted label is j.

  Returns:
    dict: A dictionary with the following keys (each value is a scalar):
      - "Macro Precision", "Macro Recall", "Macro F1", "Macro Accuracy", "Macro Specificity"
      - "Micro Precision", "Micro Recall", "Micro F1", "Micro Accuracy", "Micro Specificity"
      - "Weighted Precision", "Weighted Recall", "Weighted F1", "Weighted Accuracy", "Weighted Specificity"

  Notes:
    - Macro averaging computes metrics independently per class and then
      averages them (treats all classes equally).
    - Micro averaging aggregates contributions of all classes to compute
      the metrics (equivalent to computing metrics on the flattened
      set of predictions and labels).
    - Weighted averaging uses the number of true samples per class as
      weights when averaging per-class metrics.
    - Division by zero can occur for degenerate confusion matrices; in
      such cases NumPy will produce NaN or inf values. Callers may want
      to sanitize the confusion matrix or handle NaNs after receiving
      the results.
  '''

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
  accuracy = np.mean(TP + TN) / np.sum(cm)
  specificity = np.mean(TN / (TN + FP))

  results["Macro Precision"] = precision
  results["Macro Recall"] = recall
  results["Macro F1"] = f1
  results["Macro Accuracy"] = accuracy
  results["Macro Specificity"] = specificity

  # Using micro averaging.
  precision = np.sum(TP) / np.sum(TP + FP)
  recall = np.sum(TP) / np.sum(TP + FN)
  f1 = 2.0 * precision * recall / (precision + recall)
  accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
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
  f1 = 2.0 * precision * recall / (precision + recall)
  accuracy = np.sum((TP + TN) * weights) / np.sum(cm)
  specificity = np.sum(TN / (TN + FP) * weights)

  results["Weighted Precision"] = precision
  results["Weighted Recall"] = recall
  results["Weighted F1"] = f1
  results["Weighted Accuracy"] = accuracy
  results["Weighted Specificity"] = specificity

  return results


def PretrainedModelHyperparamsBuilderKT():
  hp = kt.HyperParameters()

  # Which pretrained backbone to use for transfer learning / fine-tuning.
  hp.Choice("baseModel", ["MobileNetV2", "InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"])
  # Optimizer family to use when compiling the model.
  hp.Choice("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax", "Nadam"])
  # Learning rate choices to search over.
  hp.Choice("learningRate", [1e-2, 1e-3, 1e-4, 1e-5])
  # Batch size options used during training.
  hp.Choice("batchSize", [8, 16, 32, 64])
  # Whether to insert dropout layers in the classifier head.
  hp.Choice("applyDropout", [True, False])
  # Dropout rate options when dropout is applied.
  hp.Choice("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])

  return hp


def PretrainedModelBuilderKT(inputShape=(256, 256, 3), noOfClasses=4):
  def _helper(hp):
    baseModelCls = {
      "MobileNetV2": MobileNetV2,
      "InceptionV3": InceptionV3,
      "ResNet50"   : ResNet50,
      "VGG16"      : VGG16,
      "VGG19"      : VGG19,
      "Xception"   : Xception,
    }

    optimizerCls = {
      "Adam"    : Adam,
      "RMSprop" : RMSprop,
      "SGD"     : SGD,
      "Adagrad" : Adagrad,
      "Adadelta": Adadelta,
      "Adamax"  : Adamax,
      "Nadam"   : Nadam,
    }

    # Instantiate the selected base model with pretrained ImageNet weights and
    # without the top classification layer.
    selectedModel = hp["baseModel"]
    baseModel = baseModelCls[selectedModel](
      # Exclude default classification head; we will add a custom head.
      include_top=False,
      # Initialize with ImageNet weights.
      weights="imagenet",
      # Input image shape for the model.
      input_shape=inputShape,
    )

    # Freeze the base model layers to prevent them from being updated during training.
    for layer in baseModel.layers:
      layer.trainable = False

    layers = [
      baseModel,
      # Pool spatial features and produce a vector representation.
      GlobalAveragePooling2D(),
      # Dense layers for the custom classification head.
      Dense(128, activation="relu"),
      Dense(64, activation="relu"),
      # Final classification layer.
      Dense(noOfClasses, activation="softmax") if (noOfClasses > 2) else Dense(1, activation="sigmoid")
    ]

    if (hp["applyDropout"]):
      # Insert dropout layers into the head to help regularize training.
      layers.insert(4, Dropout(hp["dropout"]))
      layers.insert(3, Dropout(hp["dropout"]))

    model = Sequential(layers)

    selectedOptimizer = hp["optimizer"]
    selectedLR = hp["learningRate"]
    model.compile(
      # Instantiate the optimizer selected by the tuner with the chosen LR.
      optimizer=optimizerCls[selectedOptimizer](learning_rate=selectedLR),
      # Use categorical cross-entropy for multi-class classification.
      loss="categorical_crossentropy" if (noOfClasses > 2) else "binary_crossentropy",
      # Useful metrics to monitor during training and evaluation.
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

    # Print the model summary to visualize the architecture and number of parameters.
    model.summary()

    # Return the compiled model to the tuner for training with the current set of hyperparameters.
    return model

  # The outer function returns the inner helper function which takes the hyperparameters
  # as input and builds the model accordingly.
  return _helper


def PretrainedModelKerasTuner(
  inputShape=(256, 256, 3),
  maxEpochs=100,
  noOfClasses=4,
  directory="History",
  projectName="PretrainedKerasTuner",
):
  hp = PretrainedModelHyperparamsBuilderKT()

  # Instantiate the Hyperband tuner which will perform an efficient hyperparameter search over the defined space.
  tuner = kt.Hyperband(
    PretrainedModelBuilderKT(inputShape=inputShape, noOfClasses=noOfClasses),
    hyperparameters=hp,
    objective="val_categorical_accuracy",
    max_epochs=maxEpochs,  # This is the maximum number of epochs to train the model.
    factor=7,  # This is the downsampling factor for the number of epochs.
    directory=directory,
    project_name=projectName,
  )

  return tuner


def PretrainedModelOptuna(
  baseModelStr,
  optimizerStr,
  dropout,
  applyDropout,
  learningRate,
  inputShape=(256, 256, 3),
  noOfClasses=4,
):
  baseModelCls = {
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "ResNet50"   : ResNet50,
    "VGG16"      : VGG16,
    "VGG19"      : VGG19,
    "Xception"   : Xception,
  }

  optimizerCls = {
    "Adam"    : Adam,
    "RMSprop" : RMSprop,
    "SGD"     : SGD,
    "Adagrad" : Adagrad,
    "Adadelta": Adadelta,
    "Adamax"  : Adamax,
    "Nadam"   : Nadam,
  }

  # Instantiate the selected base model with pretrained ImageNet weights and
  # without the top classification layer.
  baseModel = baseModelCls[baseModelStr](
    # Exclude default classification head; we will add a custom head.
    include_top=False,
    # Initialize with ImageNet weights.
    weights="imagenet",
    # Input image shape for the model.
    input_shape=inputShape,
  )

  # Freeze the base model layers to prevent them from being updated during training.
  for layer in baseModel.layers:
    layer.trainable = False

  layers = [
    baseModel,
    # Pool spatial features and produce a vector representation.
    GlobalAveragePooling2D(),
    # Dense layers for the custom classification head.
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    # Final classification layer.
    Dense(noOfClasses, activation="softmax") if (noOfClasses > 2) else Dense(1, activation="sigmoid")
  ]

  if (applyDropout):
    # Insert dropout layers into the head to help regularize training.
    layers.insert(4, Dropout(dropout))
    layers.insert(3, Dropout(dropout))

  model = Sequential(layers)

  model.compile(
    # Instantiate the optimizer selected by the tuner with the chosen LR.
    optimizer=optimizerCls[optimizerStr](learning_rate=learningRate),
    # Use categorical cross-entropy for multi-class classification.
    loss="categorical_crossentropy" if (noOfClasses > 2) else "binary_crossentropy",
    # Useful metrics to monitor during training and evaluation.
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


def PretrainedModelOptunaObjectiveFunction(inputShape, trainGen, valGen, testGen, noOfClasses, maxEpochs):
  def _helper(trial):
    try:
      clear_session()

      baseModelStr = trial.suggest_categorical(
        "baseModel",
        ["MobileNetV2", "InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"]
      )

      optimizerClsStr = trial.suggest_categorical(
        "optimizer",
        ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax", "Nadam"]
      )

      dropoutRatio = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
      batchSize = trial.suggest_categorical("batchSize", [8, 16, 32, 64])
      applyDropout = trial.suggest_categorical("applyDropout", [True, False])
      learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True)

      model = PretrainedModelOptuna(
        baseModelStr,
        optimizerClsStr,
        dropoutRatio,
        applyDropout,
        learningRate,
        inputShape,
        noOfClasses,
      )

      model.fit(
        trainGen,
        epochs=maxEpochs,
        validation_data=valGen,
        batch_size=batchSize,
        callbacks=[
          EarlyStopping(patience=3),
          ReduceLROnPlateau(factor=0.5, patience=3),
        ],
        verbose=1,
      )

      # Evaluate the model on the test set and return the categorical accuracy.
      result = model.evaluate(testGen, batch_size=batchSize, verbose=0)

      return result[1]  # Return the categorical accuracy.

    except Exception as e:
      return 0.0

  return _helper


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
    # Number of attention heads.
    num_heads=configs["NumAttentionHeads"],
    # Size of each attention head for query and key.
    key_dim=configs["EmbedDimension"] // configs["NumAttentionHeads"],
    # Size of each attention head for value.
    value_dim=configs["EmbedDimension"] // configs["NumAttentionHeads"],
    # Dropout rate after attention.
    dropout=configs["DropoutRatio"],
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


class PositionEmbedding(Layer):
  def __init__(self, numPatches, embedDim, **kwargs):
    super().__init__(**kwargs)
    self.posEmbedding = Embedding(input_dim=numPatches, output_dim=embedDim)
    self.positions = tf.range(start=0, limit=numPatches, delta=1)

  def call(self, inputs):
    # Broadcasting handles batch dimension automatically.
    return self.posEmbedding(self.positions)


def ClassificationHead(z, configs):
  z = Dense(configs["EmbedDimension"], activation=configs["HiddenActivation"])(z)
  z = Dropout(configs["DropoutRatio"])(z)
  z = Dense(configs["NumClasses"], activation=configs["OutputActivation"])(z)
  return z


def BasicVisionTransformer(configs):
  inputShape = (configs["NumPatches"], configs["EmbedDimension"])
  inputs = Input(inputShape)

  patchEmbed = PatchEmbedding(inputs, configs)  # Create the patch embeddings.
  # Create the position embeddings.
  posEmbed = PositionEmbedding(configs["NumPatches"], configs["EmbedDimension"])(patchEmbed)
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
  output = ClassificationHead(cls, configs)

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


def ImageToViTPatches(image, noOfPatches, patchSize):
  patches = patchify.patchify(image, (patchSize, patchSize, 3), step=patchSize)
  patches = patches.reshape(-1, patchSize, patchSize, 3)
  patches = patches[:noOfPatches]
  patches = patches.reshape(-1, patchSize * patchSize * 3)
  return patches


class ViTPatchDataGeneratorFromFolder(Sequence):

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
      patches = ImageToViTPatches(image, self.noOfPatches, self.patchSize)

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

  trainGen = ViTPatchDataGeneratorFromFolder(trainFolder, inputShape, batchSize, shuffle=True, **kwargs)
  valGen = ViTPatchDataGeneratorFromFolder(valFolder, inputShape, batchSize, **kwargs)
  testGen = ViTPatchDataGeneratorFromFolder(testFolder, inputShape, batchSize, **kwargs)

  return trainGen, valGen, testGen


class ViTPatchDataGeneratorFromDataFrame(Sequence):
  '''
  Keras Sequence generator for Vision Transformer training using a pandas DataFrame.
  This generator loads images from file paths specified in a DataFrame, extracts
  non-overlapping patches, flattens them to embedding vectors, and yields batches 
  compatible with the VisionTransformer model architecture.
  '''

  def __init__(
    self, dataFrame, inputShape, batchSize, classMode="categorical",
    noOfPatches=256, patchSize=16, embedDimension=768, shuffle=False,
  ):
    # Validate required DataFrame columns.
    requiredCols = {"image_path", "label"}
    if (not requiredCols.issubset(dataFrame.columns)):
      raise ValueError(f"DataFrame must contain columns: {requiredCols}")

    self.dataFrame = dataFrame.reset_index(drop=True)
    self.inputShape = inputShape
    self.batchSize = batchSize
    self.classMode = classMode
    self.noOfPatches = noOfPatches
    self.patchSize = patchSize
    self.embedDimension = embedDimension
    self.shuffle = shuffle

    # Build class mapping for categorical encoding.
    self.classes = sorted(dataFrame["label"].unique())
    self.classIndices = {label: idx for idx, label in enumerate(self.classes)}
    self.numClasses = len(self.classes)
    self.yTrue = dataFrame["label"].values
    self.yTrueIndices = np.array([self.classIndices[label] for label in self.yTrue])

    self.numImages = len(self.dataFrame)
    self.indices = np.arange(self.numImages)

    # Warn about dropped samples if batch size does not divide evenly.
    remainder = self.numImages % self.batchSize
    if (remainder != 0):
      print(
        f"Warning: {remainder} samples will be dropped per epoch "
        f"(batchSize={batchSize}, total={self.numImages})"
      )
      self.yTrue = self.yTrue[:-remainder]
      self.yTrueIndices = self.yTrueIndices[:-remainder]

    if (self.shuffle):
      np.random.shuffle(self.indices)

  def __len__(self):
    '''Return number of batches per epoch.'''
    return self.numImages // self.batchSize

  def _image_to_patches(self, image):
    '''Extract and flatten non-overlapping patches from a preprocessed image.'''

    # Extract patches using patchify: returns (grid_h, grid_w, 1, ph, pw, c).
    patches = patchify.patchify(image, (self.patchSize, self.patchSize, 3), step=self.patchSize)
    # Reshape to flat list: (num_patches, patchSize, patchSize, 3)
    patches = patches.reshape(-1, self.patchSize, self.patchSize, 3)
    # Truncate or pad to expected number of patches
    if (len(patches) < self.noOfPatches):
      # Pad with zeros if insufficient patches (edge case for non-divisible dimensions).
      padding = np.zeros((self.noOfPatches - len(patches), self.patchSize, self.patchSize, 3))
      patches = np.concatenate([patches, padding], axis=0)
    else:
      patches = patches[:self.noOfPatches]
    # Flatten each patch to vector: (noOfPatches, patchSize*patchSize*3).
    return patches.reshape(self.noOfPatches, -1)

  def __getitem__(self, index):
    '''Generate one batch of data.'''
    # Select batch indices.
    batchIndices = self.indices[index * self.batchSize: (index + 1) * self.batchSize]

    # Pre-allocate batch arrays (use empty for slight efficiency gain)
    images = np.empty((self.batchSize, self.noOfPatches, self.embedDimension), dtype=np.float32)

    if (self.classMode == "categorical"):
      labels = np.zeros((self.batchSize, self.numClasses), dtype=np.float32)
    else:
      labels = np.zeros((self.batchSize, 1), dtype=np.int32)

    for i, idx in enumerate(batchIndices):
      # Load image path and label from DataFrame.
      row = self.dataFrame.iloc[idx]
      imgPath = row["image_path"]
      label = row["label"]

      # Load and validate image.
      image = cv2.imread(imgPath)
      if (image is None):
        raise ValueError(f"Failed to load image: {imgPath}")

      # Convert BGR (OpenCV default) to RGB for model compatibility.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Resize to expected input dimensions.
      image = cv2.resize(
        image,
        (self.inputShape[1], self.inputShape[0]),  # (width, height) for cv2.resize.
        interpolation=cv2.INTER_CUBIC
      )

      # Extract and flatten patches.
      patches = self._image_to_patches(image)
      images[i] = patches

      # Encode label.
      if (self.classMode == "categorical"):
        labels[i, self.classIndices[label]] = 1.0
      else:
        labels[i, 0] = self.classIndices[label]

    # Normalize pixel values to [0, 1].
    return images / 255.0, labels

  def on_epoch_end(self):
    '''Callback invoked at the end of each epoch for shuffling.'''
    if self.shuffle:
      np.random.shuffle(self.indices)

  def get_class_indices(self):
    '''Return the mapping from class names to integer indices.'''
    return self.classIndices.copy()


def PretrainedVisionTransformer(
  datasetDir, modelName, outputDir, applyDataAugmentation=True, testSize=0.15,
  numTrainEpochs=32, batchSize=16, learningRate=2e-4, fp16=True, saveSteps=25,
  loggingSteps=10,
):
  # ========================================================================
  # PREPROCESSING FUNCTIONS
  # ========================================================================
  # The Trainer expects batches with keys `pixel_values` and `labels`. We define
  # small helper functions for transforming images for training/validation,
  # collating batches for the Trainer, and computing evaluation metrics.
  def _PreprocessTrain(exampleBatch):
    # Convert the images to RGB and apply the training transforms. The
    # resulting `pixel_values` key contains tensors the model expects.
    exampleBatch["pixel_values"] = [
      trainTransforms(image.convert("RGB")) for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _PreprocessVal(exampleBatch):
    # Convert the images to RGB and apply the validation transforms. The
    # resulting `pixel_values` key contains tensors the model expects.
    exampleBatch["pixel_values"] = [
      valTransforms(image.convert("RGB")) for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _CollateFunc(batch):
    # Collate function used by the Trainer: stack pixel tensors and convert
    # labels to a torch tensor. Returned dict matches the model's input API.
    pixelValues = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixelValues, "labels": labels}

  def _ComputeMetrics(evalPred):
    # Compute predictions -> confusion matrix -> extended metrics using the
    # project's helper `CalculateAllMetrics` to keep evaluation consistent.
    predictions = np.argmax(evalPred.predictions, axis=1)
    references = evalPred.label_ids
    cm = confusion_matrix(references, predictions)
    metrics = CalculateAllMetrics(cm)
    return metrics

  # ========================================================================
  # FEATURE EXTRACTOR AND TRANSFORMS
  # ========================================================================
  # Load the ViT feature extractor to obtain the expected input size and the
  # normalization parameters. These are used to build consistent torchvision
  # transforms for training and validation.
  featureExtractor = ViTImageProcessor.from_pretrained(modelName)
  feSize = (featureExtractor.size.get("height"), featureExtractor.size.get("width"))
  normalize = Normalize(mean=featureExtractor.image_mean, std=featureExtractor.image_std)

  # Build torchvision transforms for training and validation. When augmentation
  # is enabled we use RandomResizedCrop + horizontal flip, otherwise deterministic
  # Resize + CenterCrop to match validation behavior.
  if (applyDataAugmentation):
    trainTransforms = Compose([
      RandomResizedCrop(feSize),
      RandomHorizontalFlip(),
      ToTensor(),
      normalize,
    ])
  else:
    trainTransforms = Compose([
      Resize(feSize),
      CenterCrop(feSize),
      ToTensor(),
      normalize,
    ])

  valTransforms = Compose([
    Resize(feSize),
    CenterCrop(feSize),
    ToTensor(),
    normalize,
  ])

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
    id2label=id2Label,  # Define the ID to label mapping.
    label2id=label2ID,  # Define the label to ID mapping.
    ignore_mismatched_sizes=True,  # Ignore mismatched sizes.
  )

  # Define the training arguments.
  trainingArgs = TrainingArguments(
    output_dir=outputDir,  # Define the output directory.
    per_device_train_batch_size=batchSize,  # Define the batch size.
    eval_strategy="steps",  # Define the evaluation strategy.
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
    processing_class=featureExtractor,  # Define the tokenizer.
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
  # ========================================================================
  # MODEL & FEATURE EXTRACTOR LOADING
  # ========================================================================
  # Load the fine-tuned model and its corresponding feature extractor from the
  # specified output directory. The `imagesBasePath` directory is expected to
  # contain subfolders for the dataset split (e.g., train/val/test) and class
  # subfolders within that split.
  classes = os.listdir(os.path.join(imagesBasePath, splitType))
  featureExtractorX = AutoFeatureExtractor.from_pretrained(outputDir)
  modelX = AutoModelForImageClassification.from_pretrained(outputDir)

  results = []

  # ========================================================================
  # INFERENCE LOOP
  # ========================================================================
  # Perform batched inference with gradients disabled to reduce memory usage.
  with torch.no_grad():
    # Iterate over each ground-truth class folder and run the model on supported
    # image files found inside. We collect predicted label, probability and other
    # diagnostic information for each image.
    for cls in classes:
      clsPath = os.path.join(imagesBasePath, splitType, cls)
      files = os.listdir(clsPath)

      for i in tqdm.tqdm(range(len(files))):
        imagePath = os.path.join(clsPath, files[i])
        extension = imagePath.split(".")[-1].lower()
        if (extension not in ["png", "jpg", "bmp", "jpeg", "tiff", "tif"]):
          continue

        # Load image and prepare inputs using the feature extractor.
        image = Image.open(imagePath).convert("RGB")
        features = featureExtractorX(image, return_tensors="pt")

        # Run the model to obtain logits and derive probabilities/prediction.
        outputs = modelX(**features)
        logits = outputs.logits
        prob = logits.softmax(-1).max().item()
        probabilities = logits.softmax(-1).tolist()[0]
        predictedClassIDx = logits.argmax(-1).item()
        predictedCls = modelX.config.id2label[predictedClassIDx]

        # Append the inference record for later aggregation and saving.
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
  results = CalculateAllMetrics(cm)
  df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
  df.to_csv(os.path.join(outputDir, f"{splitType.capitalize()}_Metrics.csv"))

  # Clear the cache to avoid memory issues.
  torch.cuda.empty_cache()


def PretrainedVisionTransformerDataFrame(
  trainDF=None,  # Pandas DataFrame containing training data with columns "image_path" and "label".
  valDF=None,  # Pandas DataFrame containing validation data with columns "image_path" and "label".
  # Pandas DataFrame containing test data with columns "image_path" and "label".
  # Optional, can be used for final evaluation after training.
  testDF=None,
  modelName=None,  # Name of the pretrained ViT model to use (e.g., "google/vit-base-patch16-224").
  outputDir=None,  # Directory to save the trained model and results.
  applyDataAugmentation=True,  # Whether to apply data augmentation to the training dataset.
  numTrainEpochs=32,  # Number of epochs to train the model.
  batchSize=16,  # Batch size for training and evaluation.
  learningRate=2e-4,  # Learning rate for the optimizer.
  fp16=True,  # Enable mixed precision training for faster performance on compatible GPUs.
  saveSteps=25,  # Control how often checkpoints are saved during training.
  loggingSteps=10,  # Control how often metrics are logged during training.
):
  # Validate required inputs.
  if (trainDF is None or valDF is None):
    raise ValueError("Both 'trainDF' and 'valDF' must be provided.")

  requiredCols = {"image_path", "label"}
  for name, df in [("trainDF", trainDF), ("valDF", valDF)]:
    if (not requiredCols.issubset(df.columns)):
      raise ValueError(f"{name} must contain columns: {requiredCols}")

  # ========================================================================
  # PREPROCESSING FUNCTIONS
  # ========================================================================
  def _PreprocessTrain(exampleBatch):
    """Apply training transforms to a batch of PIL Images."""
    exampleBatch["pixel_values"] = [
      trainTransforms(image.convert("RGB"))
      for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _PreprocessVal(exampleBatch):
    """Apply validation transforms to a batch of PIL Images."""
    exampleBatch["pixel_values"] = [
      valTransforms(image.convert("RGB"))
      for image in exampleBatch["image"]
    ]
    return exampleBatch

  def _CollateFunc(batch):
    """Collate function for DataLoader: stacks pixel values and labels."""
    # Stack the pixel values and convert labels to tensors. The model expects "pixel_values" and "labels" keys.
    pixelValues = torch.stack([x["pixel_values"] for x in batch])
    # Convert labels to tensor. Assuming labels are already mapped to integer IDs in the dataset.
    labels = torch.tensor([x["label"] for x in batch])
    # Return a dictionary with the keys expected by the model.
    return {"pixel_values": pixelValues, "labels": labels}

  def _ComputeMetrics(evalPred):
    """Compute evaluation metrics from predictions."""
    # Convert model predictions to class indices and compute the confusion matrix against true labels.
    predictions = np.argmax(evalPred.predictions, axis=1)
    # references are the true labels (integer IDs) from the evaluation dataset.
    references = evalPred.label_ids
    # Compute the confusion matrix using sklearn's function, which compares the true labels with the predicted labels.
    cm = confusion_matrix(references, predictions)
    # Calculate all relevant metrics from the confusion matrix using a helper function.
    return CalculateAllMetrics(cm)

  # ========================================================================
  # FEATURE EXTRACTOR AND TRANSFORMS
  # ========================================================================
  # Load the feature extractor to get the expected input size and normalization parameters.
  featureExtractor = ViTImageProcessor.from_pretrained(modelName)
  # Extract the expected input size for the model from the feature extractor configuration.
  feSize = (featureExtractor.size.get("height"), featureExtractor.size.get("width"))

  normalize = Normalize(
    mean=featureExtractor.image_mean,  # Normalize using the feature extractor's mean and std.
    std=featureExtractor.image_std  # This ensures the input images are scaled appropriately for the pretrained model.
  )

  if (applyDataAugmentation):
    # Apply data augmentation for training: random resized crop and horizontal flip.
    trainTransforms = Compose([
      RandomResizedCrop(feSize),  # Randomly crop and resize the image to the expected input size.
      RandomHorizontalFlip(),  # Randomly flip the image horizontally with a default probability of 0.5.
      ToTensor(),  # Convert the PIL Image to a PyTorch tensor.
      normalize,  # Normalize the tensor using the mean and std from the feature extractor.
    ])
  else:
    # Use deterministic transforms for training if augmentation is not applied.
    trainTransforms = Compose([
      Resize(feSize),  # Resize the image to the expected input size.
      CenterCrop(feSize),  # Center crop the image to the expected input size.
      ToTensor(),  # Convert the PIL Image to a PyTorch tensor.
      normalize,  # Normalize the tensor using the mean and std from the feature extractor.
    ])

  # Use deterministic transforms for validation (no augmentation).
  valTransforms = Compose([
    Resize(feSize),  # Resize the image to the expected input size.
    CenterCrop(feSize),  # Center crop the image to the expected input size.
    ToTensor(),  # Convert the PIL Image to a PyTorch tensor.
    normalize,  # Normalize the tensor using the mean and std from the feature extractor.
  ])

  # ========================================================================
  # DATASET CREATION FROM DATAFRAMES
  # ========================================================================
  def _CreateHFDataset(df, labelMapping):
    """Convert pandas DataFrame to Hugging Face Dataset with consistent labels."""
    # HF Dataset expects integer labels, so we map string labels to integer IDs using the provided label mapping.
    # HF means that the "label" column in the DataFrame must be converted to integer IDs that correspond to the
    # training set's label mapping.

    # Map string labels to integer IDs using training-derived mapping.
    dfCopy = df.copy()
    dfCopy["label_id"] = dfCopy["label"].astype(str).map(labelMapping)

    # Remove any unmapped labels (should not happen with proper splits).
    if (dfCopy["label_id"].isna().any()):
      unmapped = dfCopy[dfCopy["label_id"].isna()]["label"].unique()
      raise ValueError(f"Labels not in training set: {unmapped}")

    # Create HF Dataset from DataFrame.
    hfDs = Dataset.from_pandas(
      dfCopy[["image_path", "label_id"]].rename(columns={"label_id": "label"})
    )

    # Load PIL Images into dataset.
    def _LoadImage(example):
      try:
        example["image"] = Image.open(example["image_path"]).convert("RGB")
      except Exception as e:
        raise ValueError(f"Failed to load '{example['image_path']}': {e}")
      return example

    hfDs = hfDs.map(_LoadImage, num_proc=1, desc="Loading images")
    return hfDs

  # Build label mapping from training data ONLY (ensures consistency).
  trainLabels = sorted(trainDF["label"].astype(str).unique())
  # Create label to ID and ID to label mappings based on the training labels.
  label2ID = {label: idx for idx, label in enumerate(trainLabels)}
  id2Label = {idx: label for label, idx in label2ID.items()}

  # Create HF datasets for training and validation.
  trainDS = _CreateHFDataset(trainDF, label2ID)
  valDS = _CreateHFDataset(valDF, label2ID)

  print(f"Classes ({len(trainLabels)}): {trainLabels}")
  print(f"Label mapping: {label2ID}")
  print(f"Training samples: {len(trainDS)}, Validation samples: {len(valDS)}")

  # Apply transforms to the datasets. The transforms will be applied on-the-fly during training and evaluation.
  trainDS.set_transform(_PreprocessTrain)
  valDS.set_transform(_PreprocessVal)

  # ========================================================================
  # MODEL INITIALIZATION
  # ========================================================================
  # Load the pretrained ViT model for image classification, specifying the number of labels and the label mappings.
  model = ViTForImageClassification.from_pretrained(
    modelName,  # Load the pretrained model.
    num_labels=len(trainLabels),  # Set the number of labels based on the training set.
    id2label=id2Label,  # HF requires string keys.
    label2id=label2ID,  # Map label (str) -> id (int) as required by HF.
    # Allows loading pretrained weights even if the classification head size differs from the
    # pretrained model's original head.
    ignore_mismatched_sizes=True,
  )

  # ========================================================================
  # TRAINING CONFIGURATION
  # ========================================================================
  # Define the training arguments for the Hugging Face Trainer, including output directory, batch size, number of
  # epochs, learning rate, and evaluation strategy.
  trainingArgs = TrainingArguments(
    output_dir=outputDir,  # Directory where the model checkpoints and logs will be saved.
    per_device_train_batch_size=batchSize,  # Batch size for training. Adjust based on GPU memory.
    eval_strategy="steps",  # Evaluate the model every "save_steps" during training.
    num_train_epochs=numTrainEpochs,  # Total number of training epochs.
    fp16=fp16,  # Use mixed precision training if True (requires compatible hardware).
    save_steps=saveSteps,  # Save a checkpoint every "save_steps" during training.
    eval_steps=saveSteps,  # Evaluate the model every "save_steps" during training.
    logging_steps=loggingSteps,  # Log training metrics every "logging_steps" during training.
    learning_rate=learningRate,  # Learning rate for the optimizer.
    save_total_limit=2,  # Maximum number of checkpoints to keep. Older checkpoints will be deleted.
    # Do not remove unused columns from the dataset (important for custom collate function).
    remove_unused_columns=False,
    push_to_hub=False,  # Do not push the model to the Hugging Face Hub.
    report_to="tensorboard",  # Report training metrics to TensorBoard for visualization.
    load_best_model_at_end=True,  # Load the best model (based on evaluation metric) at the end of training.
    metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model (lower is better).
    greater_is_better=False,  # Since we want to minimize "eval_loss", set "greater_is_better" to False.
    # Set log level to "error" to reduce verbosity (can be adjusted to "info" or "debug" for more detailed logs).
    log_level="error",
  )

  # ========================================================================
  # TRAINER INITIALIZATION AND TRAINING
  # ========================================================================
  # Initialize the Hugging Face Trainer with the model, training arguments, datasets, feature extractor
  # for tokenization, custom metric computation function, and custom data collator.
  trainer = Trainer(
    model=model,  # The model to be trained.
    args=trainingArgs,  # The training arguments defined above.
    train_dataset=trainDS,  # The training dataset created from the training DataFrame.
    eval_dataset=valDS,  # The validation dataset created from the validation DataFrame.
    # The feature extractor is used as the tokenizer for the Trainer, which will handle the preprocessing of images.
    processing_class=featureExtractor,
    # The function to compute evaluation metrics from the model's predictions during evaluation.
    compute_metrics=_ComputeMetrics,
    # The custom collate function to prepare batches of data for the model during training and evaluation.
    data_collator=_CollateFunc,
  )

  print("Starting training...")
  # Train the model using the Trainer's train method, which will handle the training loop,
  # evaluation, and checkpointing based on the defined training arguments.
  trainResults = trainer.train()

  # ========================================================================
  # SAVE MODEL AND LOG METRICS
  # ========================================================================
  # After training, save the final model to the specified output directory. The Trainer's `save_model` method
  # will save the model weights and configuration.
  os.makedirs(outputDir, exist_ok=True)
  trainer.save_model(outputDir)
  # Log the training metrics (e.g., loss, accuracy) to TensorBoard and save them to disk for later analysis.
  trainer.log_metrics("train", trainResults.metrics)
  trainer.save_metrics("train", trainResults.metrics)
  trainer.save_state()

  # Final validation evaluation and metrics logging.
  # This will evaluate the best model on the validation set and log the metrics.
  evalMetrics = trainer.evaluate()
  trainer.log_metrics("eval", evalMetrics)
  trainer.save_metrics("eval", evalMetrics)

  # Extract log history from trainer state.
  history = pd.DataFrame(trainer.state.log_history)
  trainLoss = history[history["train_loss"].notna()][["step", "train_loss"]]
  evalLoss = history[history["eval_loss"].notna()][["step", "eval_loss"]]
  plt.figure()
  # Plot training and validation loss.
  plt.plot(trainLoss["step"], trainLoss["train_loss"], label="Training Loss")
  plt.plot(evalLoss["step"], evalLoss["eval_loss"], label="Validation Loss")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  # Save the figure to the History folder for later review.
  plt.savefig(f"{outputDir}/History.png")
  # Display the plot interactively.
  # plt.show()  # Uncomment this line if you want to see the plot during execution.
  plt.close()

  # Optional: Evaluate on test set if provided and log the metrics.
  # This uses the same validation transforms for consistency.
  if (testDF is not None and len(testDF) > 0):
    print("Evaluating on test set...")
    # Create a Hugging Face Dataset from the test DataFrame using the same label mapping as the training set to
    # ensure consistency in label encoding.
    testDS = _CreateHFDataset(testDF, label2ID)
    testDS.set_transform(_PreprocessVal)  # Use validation transforms for testing.
    # Evaluate the model on the test dataset and log the metrics with a "test" prefix.
    testMetrics = trainer.evaluate(testDS, metric_key_prefix="test")
    # Log and save the test metrics separately from the training and validation metrics for clarity.
    trainer.log_metrics("test", testMetrics)
    trainer.save_metrics("test", testMetrics)
    print(f"Test metrics: {testMetrics}")

  # Memory cleanup to avoid issues in environments with limited GPU memory, especially after training and evaluation.
  torch.cuda.empty_cache()

  # Return the trainer object, evaluation metrics, and label mapping for potential further use (e.g., inference or additional evaluation).
  return trainer, evalMetrics, label2ID


def VisionTransformerInferenceDataFrame(
  # Optional pd.DataFrame with "image_path" and "label" columns for inference.
  # If None, inference will not be performed.
  testDF=None,
  # Directory containing the fine-tuned model and its configuration.
  # This is used if `modelName` is not provided to load the model for inference.
  outputDir=None,
  # Optional pretrained model name (e.g., "google/vit-base-patch16-224") to
  # load directly from Hugging Face Hub. If None, loads from `outputDir`.
  modelName=None,
):
  # Load model and feature extractor.
  if (modelName is not None):
    # If a model name is provided, load the feature extractor and model directly from the pretrained
    # model name (e.g., "google/vit-base-patch16-224").
    featureExtractorX = ViTImageProcessor.from_pretrained(modelName)
    modelX = ViTForImageClassification.from_pretrained(modelName)
  else:
    # Otherwise, load the feature extractor and model from the specified output directory,
    # which should contain the fine-tuned model and its configuration.
    featureExtractorX = ViTImageProcessor.from_pretrained(outputDir)
    modelX = AutoModelForImageClassification.from_pretrained(outputDir)

  # Set the model to evaluation mode and move it to the appropriate device (GPU if available, otherwise CPU).
  modelX.eval()
  # Determine the device to run inference on (GPU if available, otherwise CPU) and move the model to that device
  # for faster inference.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  modelX.to(device)

  # Validate input DataFrame for inference.
  # The test DataFrame must contain an "image_path" column to specify the paths of the images to be evaluated.
  # The "label" column is optional but will be used for metrics calculation if present.
  if (testDF is None or "image_path" not in testDF.columns):
    raise ValueError("`testDF` must be provided with 'image_path' column.")

  hasLabels = "label" in testDF.columns
  results = []
  validExtensions = {"png", "jpg", "bmp", "jpeg", "tiff", "tif"}

  print(f"Starting inference on {len(testDF)} images...")

  # Inference loop: iterate over each image in the test DataFrame, load and preprocess the image, run it
  # through the model to get predictions, and store the results in a structured format for later analysis.
  with torch.no_grad():
    # Use tqdm to display a progress bar for the inference loop, which provides feedback on the progress of
    # processing the test images.
    for idx in tqdm.tqdm(range(len(testDF)), desc="Inference"):
      # Extract the image path and actual label (if available) from the current row of the test DataFrame.
      row = testDF.iloc[idx]
      # Get the image path from the "image_path" column.
      # This path should point to the location of the image file to be evaluated.
      imgPath = row["image_path"]
      # Get the actual label from the "label" column if it exists; otherwise, set it to None. This allows for metrics
      # calculation later if labels are available, while still allowing inference to proceed without labels.
      actualLabel = row["label"] if (hasLabels) else None

      # Validate file extension to ensure only supported image formats are processed.
      # This helps avoid errors when trying to load unsupported files.
      ext = os.path.splitext(imgPath)[1].lower().lstrip(".")
      if (ext not in validExtensions):
        print(f"Skipping unsupported file: {imgPath}")
        continue

      try:
        # Load and preprocess image using the feature extractor.
        # The image is converted to RGB format to ensure compatibility with the model's expected input.
        image = Image.open(imgPath).convert("RGB")
        inputs = featureExtractorX(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through the model to get logits, which are the raw output scores for each class
        # before applying softmax.
        outputs = modelX(**inputs)
        logits = outputs.logits

        # Extract predictions and probabilities from the logits.
        # Apply softmax to convert logits to probabilities, and identify the predicted class index and its
        # associated probability.
        probabilities = logits.softmax(-1).cpu().tolist()[0]
        # Get the index of the predicted class (the one with the highest logit score) and its corresponding probability.
        predictedIdx = int(logits.argmax(-1).item())
        # Get the maximum probability for the predicted class to provide confidence information about the prediction.
        maxProb = probabilities[predictedIdx]
        # Map the predicted class index back to the class label using the model's configuration, which contains the
        # mapping from class IDs to labels. If the predicted index is not found in the mapping,
        # a default label is generated.
        predictedLabel = modelX.config.id2label[predictedIdx]

        # Store results in a structured format for later analysis.
        # This includes the image name, actual class (if available), predicted class, probability, and the
        # full list of probabilities for all classes.
        record = {
          "Image Name"        : os.path.basename(imgPath),
          "Image Path"        : imgPath,
          "Actual Class"      : actualLabel if (hasLabels) else "N/A",
          "Predicted Class ID": predictedIdx,
          "Predicted Class"   : predictedLabel,
          "Probability"       : round(float(maxProb), 4),
          "Probabilities"     : [round(float(p), 4) for p in probabilities],
        }
        results.append(record)

      except Exception as e:
        print(f"Error processing {imgPath}: {e}")
        continue

  # ========================================================================
  # SAVE AND ANALYZE RESULTS
  # ========================================================================
  if (not results):
    print("No valid images processed.")
    return None

  # Create a DataFrame from the results for easier analysis and saving.
  # This DataFrame will contain detailed information about each inference, including the predicted class,
  # actual class (if available), probabilities, and other relevant details.
  dfResults = pd.DataFrame(results)
  os.makedirs(outputDir, exist_ok=True)

  # Save detailed results to CSV for further analysis. This includes the predicted class, actual class
  # (if available), probabilities, and other relevant information for each image.
  dfResults.to_csv(
    os.path.join(outputDir, "Test_Results.csv"),
    index=False
  )
  print(f"Results saved to {os.path.join(outputDir, 'Test_Results.csv')}")

  # Compute metrics if labels are available in the test DataFrame.
  # This allows for evaluation of model performance on the test set using confusion matrix and derived metrics.
  if (hasLabels):
    # Filter out any rows with missing labels.
    # This ensures that metrics are calculated only on samples where the actual class is known.
    dfEval = dfResults[dfResults["Actual Class"] != "N/A"].copy()

    if (len(dfEval) > 0):
      references = dfEval["Actual Class"]
      predictions = dfEval["Predicted Class"]

      # Get sorted unique classes for consistent matrix ordering.
      allClasses = sorted(set(references) | set(predictions))

      # Confusion matrix calculation and visualization using sklearn's confusion_matrix and `ConfusionMatrixDisplay`.
      # The confusion matrix is saved as a high-resolution PNG file for detailed analysis.
      cm = confusion_matrix(references, predictions, labels=allClasses)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=allClasses)
      disp.plot(xticks_rotation=45, cmap="Blues")
      plt.tight_layout()
      plt.savefig(
        os.path.join(outputDir, "Test_ConfusionMatrix.png"),
        dpi=720,
        bbox_inches="tight",
      )
      plt.close()
      print(f"Confusion matrix saved to {os.path.join(outputDir, 'Test_ConfusionMatrix.png')}")

      # Calculate and save metrics derived from the confusion matrix, such as accuracy, precision, recall, and F1-score.
      metrics = CalculateAllMetrics(cm)
      dfMetrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
      dfMetrics.to_csv(os.path.join(outputDir, "Test_Metrics.csv"))
      print(f"Metrics saved to {os.path.join(outputDir, 'Test_Metrics.csv')}")

      # Print summary of metrics to console for quick reference.
      print(f"\n{'=' * 50}")
      print(f"Test Set Summary ({len(dfEval)} samples):")
      print(f"{'=' * 50}")
      for key in metrics.keys():
        if (isinstance(metrics[key], (int, float))):
          print(f"{key:12s}: {metrics[key]:.4f}")
      print(f"{'=' * 50}\n")

  # Memory cleanup to avoid issues in environments with limited GPU memory,
  # especially after inference and metrics calculation.
  torch.cuda.empty_cache()

  return dfResults
