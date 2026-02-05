'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import the openslide Python wrapper for accessing whole-slide images.
import openslide
# Import the XML ElementTree module for parsing annotation XML files.
import xml.etree.ElementTree as ET
# Import NumPy for numerical operations.
import numpy as np
# Import matplotlib.pyplot for plotting images and drawing overlays.
import matplotlib.pyplot as plt
# Import the helper function to extract BACH annotations from XML.
from HMB_Spring_2026_Helpers import ExtractBACHAnnotationsFromXML

# Define the path to the SVS file to open.
svsFile = r"Data/A02.svs"
# Define the path to the XML annotations file to parse.
xmlFile = r"Data/A02.xml"

# Open the SVS file as an OpenSlide object for reading image regions.
slide = openslide.OpenSlide(svsFile)

# Extract the list of annotation dictionaries from the XML file.
anList = ExtractBACHAnnotationsFromXML(xmlFile, verbose=False)

# Create a thumbnail of the slide for quick visualization by scaling down by 100x.
thumbnail = slide.get_thumbnail(
  (slide.dimensions[0] // 100, slide.dimensions[1] // 100)
)

# Create a new matplotlib figure for displaying the thumbnail and overlays.
plt.figure()
# Display the thumbnail image in the current axes.
plt.imshow(thumbnail)
# Disable axis drawing to show a clean image without ticks or labels.
plt.axis("off")
# Adjust the layout to remove excess whitespace around the figure.
plt.tight_layout()
# Iterate over each parsed annotation to draw polygons and labels on the thumbnail.
for an in anList:
  # Compute the x coordinates scaled to thumbnail resolution by integer division.
  x = [coord[0] // 100 for coord in an["Coords"]]
  # Compute the y coordinates scaled to thumbnail resolution by integer division.
  y = [coord[1] // 100 for coord in an["Coords"]]
  # Determine polygon color based on the annotation Text label.
  if (an["Text"].strip() == "Benign"):
    color = "g"
  elif (an["Text"].strip() == "Carcinoma in situ"):
    color = "y"
  elif (an["Text"].strip() == "Carcinoma invasive"):
    color = "r"
  else:
    # Use blue for any other annotation types.
    color = "b"  # Normal.
  # Place the annotation text at the polygon's minimum x and y coordinates.
  textLength = len(an["Text"])
  plt.text(
    np.mean(x) - textLength,  # Text x position.
    np.mean(y),  # Text y position.
    an["Text"],  # Text to display.
    fontsize=8,  # Font size for the text.
    color="black",  # Text color.
    backgroundcolor="white",  # Text background color.
  )
  # Fill the polygon with a semi-transparent color on the thumbnail.
  plt.fill(x, y, color, alpha=0.6)
  # Draw the polygon outline on top of the filled polygon.
  plt.plot(x + [x[0]], y + [y[0]], color)

# Render the figure window and block until it is closed by the user.
plt.show()
# Close the current figure and free its associated resources.
plt.close()

# Close the SVS file to free associated resources.
slide.close()
