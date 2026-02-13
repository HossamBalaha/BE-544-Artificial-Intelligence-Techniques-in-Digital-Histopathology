'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import openslide for reading whole-slide images (WSIs) in SVS format.
import openslide
# Import matplotlib pyplot for plotting images and figures.
import matplotlib.pyplot as plt
# Import helper functions for parsing and mapping annotations from the helpers module.
from HMB_Spring_2026_Helpers import (
  ExtractBACHAnnotationsFromXML,
  MapAnnotationsFromLevelAToLevelB,
  ExtractWSIROI,
)

# Path to the SVS file inside the Data folder.
svsFile = r"Data/A02.svs"
# Path to the corresponding XML annotation file inside the Data folder.
xmlFile = r"Data/A02.xml"

# Open the SVS file using OpenSlide to access pyramid levels.
slide = openslide.OpenSlide(svsFile)

# Query the number of pyramid levels in the slide.
slideLevels = slide.level_count

# Define the requested output region size in pixels.
(width, height) = (512, 512)
# Set the annotation index to display or process.
annotationIndex = 0  # Index of the annotation.

# Create a new matplotlib figure for plotting results.
plt.figure()

# Iterate through each level in the slide pyramid to extract and display views.
for level in range(slideLevels):
  # Calculate the downsample ratio with reference to the highest resolution level.
  dRatio = int(slide.level_downsamples[level] / slide.level_downsamples[0])

  # Extract BACH annotations from the XML file at the base level.
  anList = ExtractBACHAnnotationsFromXML(xmlFile, verbose=False)

  # Select the annotation dictionary using the specified index.
  region = anList[annotationIndex]
  # Map the annotation coordinates from level A to the current level B using helper function.
  mappedRegion = MapAnnotationsFromLevelAToLevelB(region, dRatio)

  (regionSlide, regionMask, roi) = ExtractWSIROI(
    # An OpenSlide object representing the whole-slide image (WSI) or an object with a `read_region` method and `dimensions` attribute.
    slide,
    # A dictionary with a "Coords" key containing a list of (x, y) tuples representing the vertices of the polygon.
    region=mappedRegion,
    level=level,  # The pyramid level to read the region from (default is 0 for highest resolution).
    dRatio=dRatio,  # The downsample ratio to apply to the coordinates (default is 1.0 for no change).
    width=width,  # The width of the output region to extract in pixels (default is 512).
    height=height,  # The height of the output region to extract in pixels (default is 512).
    startX=0,  # The x-coordinate offset to apply to the region when reading from the slide (default is 0).
    startY=0,  # The y-coordinate offset to apply to the region when reading from the slide (default is 0).
    # The minimum percentage of the mask that must be present in the extracted region to consider it valid (default is 0.0 for no threshold).
    maskThreshold=0.0,
  )

  # Select the subplot location and display the full region image.
  plt.subplot(3, slideLevels, level + 1)
  # Show the region image in the subplot.
  plt.imshow(regionSlide)
  # Disable axis ticks and labels for the image subplot.
  plt.axis("off")
  # Adjust subplot layout to minimize overlaps.
  plt.tight_layout()
  # Set the title for the region subplot with level and downsample factor.
  plt.title(f"Level {level} ({dRatio}x)")

  # Select the subplot location and display the region mask.
  plt.subplot(3, slideLevels, 3 + level + 1)
  # Show the binary mask in grayscale in the subplot.
  plt.imshow(regionMask, cmap="gray")
  # Disable axis ticks and labels for the mask subplot.
  plt.axis("off")
  # Adjust layout for the mask subplot.
  plt.tight_layout()
  # Set the title for the mask subplot indicating the level.
  plt.title(f"Level {level} Mask")

  # Select the subplot location and display the masked ROI.
  plt.subplot(3, slideLevels, 6 + level + 1)
  # Show the ROI image in the subplot.
  plt.imshow(roi, cmap="gray")
  # Disable axis ticks and labels for the ROI subplot.
  plt.axis("off")
  # Adjust layout for the ROI subplot.
  plt.tight_layout()
  # Set the title for the ROI subplot indicating the level.
  plt.title(f"Level {level} ROI")

# Display the figure with all subplots.
plt.show()

# Close the current figure and free its associated resources.
plt.close()

# Close the SVS file to free-associated resources.
slide.close()
