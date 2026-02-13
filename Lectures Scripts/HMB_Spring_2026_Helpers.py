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
import os, cv2, openslide
import numpy as np
import xml.etree.ElementTree as ET


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


def ExtractWSIRegion(slide, region, level=0):
  r'''
  Extract a region of interest (ROI) from a whole-slide image (WSI) using OpenSlide.

  Parameters:
    slide (openslide.OpenSlide or slide-like): The OpenSlide object representing the WSI or an object with a `read_region` method and `dimensions` attribute.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples.
    level (int): The pyramid level to read the region from (default is 0 for highest resolution).

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
    level,
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


def MapAnnotationsFromLevelAToLevelB(annotation, dFactor=1.0):
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

  # Return a new annotation dictionary with the same text and mapped coordinates.
  return {
    "Text"  : annotation.get("Text", ""),
    "Coords": mappedCoords
  }


def BuildMaskFromPolygonCoords(regionCoords, padFactor=1.0):
  r'''
  Build a binary mask from polygon coordinates and apply padding.

  Parameters:
    regionCoords (list): A list of (x, y) tuples representing the vertices of the polygon.
    padFactor (float): A factor to determine how much padding to add around the region
      (default is 1.0 for no padding, 2.0 for doubling the size of the bounding box).

  Returns:
    tuple: A tuple containing:
      - regionMask (numpy.ndarray): The binary mask for the region as a NumPy uint8 array (H,W) with 0/255 values.
      - regionWidth (int): The width of the original bounding box of the region before padding.
      - regionHeight (int): The height of the original bounding box of the region before padding.
      - regionX (list): The list of original x coordinates of the polygon vertices.
      - regionY (list): The list of original y coordinates of the polygon vertices.
      - regionXShifted (list): The list of x coordinates shifted to account for padding.
      - regionYShifted (list): The list of y coordinates shifted to account for padding.
  '''

  # Extract the x coordinates from the polygon vertices.
  regionX = [x for x, y in regionCoords]
  # Extract the y coordinates from the polygon vertices.
  regionY = [y for x, y in regionCoords]
  # Compute the width of the bounding box of the region.
  regionWidth = max(regionX) - min(regionX)
  # Compute the height of the bounding box of the region.
  regionHeight = max(regionY) - min(regionY)

  # Mask the region with a contour in black and white.
  regionXShifted = [x - min(regionX) for x in regionX]
  # Shift the y coordinates so the polygon starts at zero.
  regionYShifted = [y - min(regionY) for y in regionY]
  # Pair the shifted coordinates together as integer tuples.
  regionCoordsShifted = [(x, y) for x, y in zip(regionXShifted, regionYShifted)]
  # Convert the coordinate list to a NumPy integer array for OpenCV.
  regionCoordsShifted = np.array(regionCoordsShifted, np.int32)
  # Create an empty mask of zeros with the computed bounding box size.
  regionMask = np.zeros((regionHeight, regionWidth))
  # Fill the polygon on the mask with white (255) using OpenCV.
  regionMask = cv2.fillPoly(regionMask, [regionCoordsShifted], 255)
  # Convert the mask to unsigned 8-bit integers.
  regionMask = regionMask.astype(np.uint8)

  # Compute the additional width needed to pad the region according to padFactor.
  remainingWidth = regionWidth * padFactor - regionWidth
  # Compute the additional height needed to pad the region according to padFactor.
  remainingHeight = regionHeight * padFactor - regionHeight

  # Add a constant border around the mask to implement padding.
  regionMask = cv2.copyMakeBorder(
    regionMask,
    remainingHeight // 2,
    remainingHeight - remainingHeight // 2,
    remainingWidth // 2,
    remainingWidth - remainingWidth // 2,
    cv2.BORDER_CONSTANT,
    value=0
  )

  # Shift the x coordinates to account for the added left border.
  regionXShifted = [x + remainingWidth // 2 for x in regionXShifted]
  # Shift the y coordinates to account for the added top border.
  regionYShifted = [y + remainingHeight // 2 for y in regionYShifted]

  # Return the mask and computed geometry and shifted coordinate lists.
  return (
    regionMask, regionWidth, regionHeight,
    regionX, regionY, regionXShifted, regionYShifted
  )


def ExtractWSIROI(
  # An OpenSlide object representing the whole-slide image (WSI) or an object with a `read_region` method and `dimensions` attribute.
  slide,
  # A dictionary with a "Coords" key containing a list of (x, y) tuples representing the vertices of the polygon.
  region,
  level=0,  # The pyramid level to read the region from (default is 0 for highest resolution).
  dRatio=1.0,  # The downsample ratio to apply to the coordinates (default is 1.0 for no change).
  width=512,  # The width of the output region to extract in pixels (default is 512).
  height=512,  # The height of the output region to extract in pixels (default is 512).
  startX=0,  # The x-coordinate offset to apply to the region when reading from the slide (default is 0).
  startY=0,  # The y-coordinate offset to apply to the region when reading from the slide (default is 0).
  # The minimum percentage of the mask that must be present in the extracted region to consider it valid (default is 0.0 for no threshold).
  maskThreshold=0.0,
):
  r'''
  Extract a region of interest (ROI) from a whole-slide image (WSI) using OpenSlide, given a polygon annotation.

  Parameters:
    slide (openslide.OpenSlide or slide-like): The OpenSlide object representing the WSI or an object with a `read_region` method and `dimensions` attribute.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples representing the vertices of the polygon.
    level (int): The pyramid level to read the region from (default is 0 for highest resolution).
    dRatio (float): The downsample ratio to apply to the coordinates (default is 1.0 for no change).
    width (int): The width of the output region to extract in pixels (default is 512).
    height (int): The height of the output region to extract in pixels (default is 512).
    startX (int): The x-coordinate offset to apply to the region when reading from the slide (default is 0).
    startY (int): The y-coordinate offset to apply to the region when reading from the slide (default is 0).
    maskThreshold (float): The minimum percentage of the mask that must be present in the extracted region to
      consider it valid (default is 0.0 for no threshold).

  Notes:
    If the percentage of the mask in the extracted region is below this threshold,
      the function will return None for all outputs, indicating that the region does
      not contain enough of the annotated area to be considered valid.
      This can be used to filter out regions that do not sufficiently overlap with the annotation,
      especially when extracting smaller tiles from a larger annotated area.
      For example, if `maskThreshold` is set to 0.5, then at least 50% of the pixels in the extracted
      region must be part of the mask (i.e., have a value of 255) for the function to return the
      region image, mask, and ROI. If the extracted region contains less than 50% of the mask,
      the function will return None for all outputs, indicating that the region is not valid
      for further analysis or display.

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

  # Extract the list of coordinates from the mapped annotation.
  regionCoords = region["Coords"]

  # Create the mask and compute related geometry by calling BuildMaskFromPolygonCoords.
  # regionX, regionY are the original coordinates of the annotation polygon at the current level.
  # regionXShifted, regionYShifted are the coordinates shifted to align with the requested region for cropping.
  (
    mask, regionWidth, regionHeight,
    regionX, regionY, regionXShifted, regionYShifted
  ) = BuildMaskFromPolygonCoords(
    regionCoords,
    padFactor=dRatio,
  )

  # Calculate the width padding factor to center crops at lower resolutions.
  factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
  # Calculate the height padding factor to center crops at lower resolutions.
  factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

  print(f"Level {level}: Downsample Ratio = {dRatio}x, Region Width = {regionWidth}, Region Height = {regionHeight}")
  print(f"Level {level}: Factor Width = {factorWidth}, Factor Height = {factorHeight}")

  # Compute the minimum x-coordinate of the shifted region for mask placement.
  xMask = np.min(regionXShifted) + startX
  # Compute the minimum y-coordinate of the shifted region for mask placement.
  yMask = np.min(regionYShifted) + startY
  # Adjust the mask x-coordinate by the width padding factor.
  xNewMask = xMask - factorWidth
  # Adjust the mask y-coordinate by the height padding factor.
  yNewMask = yMask - factorHeight
  # Crop the mask to the requested output size for visualization.
  regionMask = mask[
    yNewMask:yNewMask + height,  # Crop the mask vertically from yNewMask to yNewMask + height.
    xNewMask:xNewMask + width  # Crop the mask horizontally from xNewMask to xNewMask + width.
  ]

  if (regionMask.shape[0] != height) or (regionMask.shape[1] != width):
    print(
      f"Warning: Cropped mask shape {regionMask.shape} does not match requested size ({height}, {width}). "
      f"Skipping this region."
    )
    return None, None, None

  if (maskThreshold > 0.0):
    rhs = (maskThreshold / dRatio ** 2) * regionMask.size * 255
    # If the percentage of the mask is below the threshold, return None for all outputs.
    if (np.sum(regionMask) < rhs):
      print(
        f"Mask percentage is below the threshold of {maskThreshold:.4f}. "
        f"Skipping this region."
      )
      return None, None, None

  # Compute the reference x coordinate for the region in the current level.
  xRegion = dRatio * (np.min(regionX)) + startX
  # Compute the reference y coordinate for the region in the current level.
  yRegion = dRatio * (np.min(regionY)) + startY
  # Adjust the region x-coordinate by downsample and padding.
  xNewRegion = xRegion - dRatio * factorWidth
  # Adjust the region y-coordinate by downsample and padding.
  yNewRegion = yRegion - dRatio * factorHeight

  # Read the requested region from the slide at the current level.
  regionSlide = slide.read_region(
    (xNewRegion, yNewRegion),
    level,
    (width, height),
  )
  # Convert the returned PIL image to RGB color space.
  regionSlide = regionSlide.convert("RGB")
  # Convert the PIL image to a NumPy array for display and processing.
  regionSlide = np.array(regionSlide)

  # Apply the binary mask to the region using a bitwise AND operation.
  roi = cv2.bitwise_and(regionSlide, regionSlide, mask=regionMask)
  # Convert RGBA images to RGB color space after masking.
  roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

  # Return a tuple containing the region image, mask, and extracted ROI.
  return regionSlide, regionMask, roi
