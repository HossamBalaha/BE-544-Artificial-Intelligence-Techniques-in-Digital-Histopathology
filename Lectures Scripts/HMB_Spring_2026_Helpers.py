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
import os, cv2, openslide, tqdm
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon


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
    'entropy'        : entropyValue,
    'colorVariance'  : colorVariance,
    'tissueRatio'    : tissueRatio,
    'meanSaturation' : meanSaturation,
    'textureVariance': textureVariance,
    'backgroundScore': backgroundScore,
  }
