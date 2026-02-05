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
  '''
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
  '''
  Extract a region of interest (ROI) from a whole-slide image (WSI) using OpenSlide.

  Parameters:
    slide (openslide.OpenSlide): The OpenSlide object representing the WSI.
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
    0,  # Level 0.
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
