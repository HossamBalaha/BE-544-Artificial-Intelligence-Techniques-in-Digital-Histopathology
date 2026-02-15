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
# Import helper functions for parsing and mapping annotations from the helpers module.
from HMB_Spring_2026_Helpers import (
  ExtractBACHAnnotationsFromXML,
  ExtractRegionTiles
)

# Path to the SVS file inside the Data folder.
svsFile = r"Data/A02.svs"
# Path to the corresponding XML annotation file inside the Data folder.
xmlFile = r"Data/A02.xml"

# Define the requested output region size in pixels.
(width, height) = (512, 512)
# Define the overlap between neighboring tiles in pixels.
overlapWidth = 256  # Horizontal overlap (set to 0 for no overlap).
overlapHeight = 256  # Vertical overlap (set to 0 for no overlap).

# Open the SVS file using OpenSlide to access pyramid levels.
slide = openslide.OpenSlide(svsFile)

# Extract BACH annotations from the XML file at the base level.
anList = ExtractBACHAnnotationsFromXML(xmlFile, verbose=False)

for region in anList[14:]:
  # Extract the region tiles from the whole-slide image based on the selected annotation and specified parameters.
  ExtractRegionTiles(
    slide,  # OpenSlide object representing the whole-slide image.
    region,  # Dictionary containing the annotation details, including polygon coordinates.
    width=width,  # Width of the output region in pixels.
    height=height,  # Height of the output region in pixels.
    overlapWidth=overlapWidth,  # Horizontal overlap between neighboring tiles in pixels (set to 0 for no overlap).
    overlapHeight=overlapHeight,  # Vertical overlap between neighboring tiles in pixels (set to 0 for no overlap).
    storageDir="Output/A02_Tiles",
  )

print("Extraction completed.")
