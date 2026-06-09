'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import the OpenSlide library for reading whole-slide images.
import openslide
# Import matplotlib pyplot for plotting images and figures.
import matplotlib.pyplot as plt
from HMB_Spring_2026_Helpers import ExtractPyramidalWSITiles

# Path to the SVS file to open.
svsFile = r"Data/A02.svs"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Set the center coordinates of the region to extract at the base level.
(x, y) = (19000, 25000)
# Set the output region size in pixels.
(width, height) = (512, 512)

# Extract the tiles from the specified region and get the figure to return for display.
tiles, fig = ExtractPyramidalWSITiles(
  slide,
  x=x,
  y=y,
  width=width,
  height=height,
)

# Display the figure with all subplots.
plt.show()

# Close the figure to free-associated resources.
plt.close(fig)

# Close the SVS file to free-associated resources.
slide.close()
