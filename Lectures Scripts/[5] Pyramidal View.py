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
# Import NumPy for efficient numerical array operations.
import numpy as np

# Path to the SVS file to open.
svsFile = r"Data/A02.svs"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Get the slide width and height in pixels.
slideWidth, slideHeight = slide.dimensions
# Get the number of pyramid levels in the slide.
slideLevels = slide.level_count

# Set the center coordinates of the region to extract at the base level.
(x, y) = (19000, 25000)
# Set the output region size in pixels.
(width, height) = (512, 512)

print(f"Slide Dimensions: {slideWidth}x{slideHeight}, Levels: {slideLevels}")
print(f"Extracting region centered at ({x}, {y}) with size ({width}x{height}) from slide.")

# Create a new matplotlib figure to plot the regions from each level.
plt.figure()
# Iterate over each level in the slide pyramid.
for i in range(slideLevels):
  # Calculate the downsample ratio relative to the highest resolution level.
  dRatio = int(slide.level_downsamples[i] / slide.level_downsamples[0])

  # Calculate the horizontal offset factor used to center the crop at lower resolutions.
  factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
  # Calculate the vertical offset factor used to center the crop at lower resolutions.
  factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

  # Compute the new x-coordinate for the region at the current level.
  xNew = x - dRatio * factorWidth
  # Compute the new y-coordinate for the region at the current level.
  yNew = y - dRatio * factorHeight

  print(f"Level {i}: Downsample Ratio={dRatio}, xNew={xNew}, yNew={yNew}")

  # Read a region from the slide at the given level and coordinates.
  regionSlide = slide.read_region(
    (xNew, yNew),
    i,
    (width, height),
  )
  # Convert the returned PIL image to RGB mode.
  regionSlide = regionSlide.convert("RGB")
  # Convert the PIL image to a NumPy array for manipulation and display.
  regionSlide = np.array(regionSlide)

  # Select the subplot for displaying the full region at this pyramid level.
  plt.subplot(2, slideLevels, i + 1)
  # Render the region image in the subplot.
  plt.imshow(regionSlide)
  # Disable axis ticks and labels for the image subplot.
  plt.axis("off")
  # Adjust subplot layout to minimize overlaps.
  plt.tight_layout()
  # Set a title for the subplot including level and downsample factor.
  plt.title(f"Level {i} ({dRatio}x)")

  # Crop the rendered region to verify the corresponding area at the current level.
  verify = regionSlide[
    factorHeight:factorHeight + height // dRatio,
    factorWidth:factorWidth + width // dRatio,
  ]

  # Select the subplot for displaying the verification crop.
  plt.subplot(2, slideLevels, 3 + i + 1)
  # Render the verification crop in the subplot.
  plt.imshow(verify)
  # Disable axis ticks and labels for the verification subplot.
  plt.axis("off")
  # Adjust layout for the verification subplot.
  plt.tight_layout()
  # Set a title for the verification subplot indicating the level verified.
  plt.title(f"Cropped to Verify Level {i}")

# Display the figure with all subplots.
plt.show()

# Close the current figure and free its associated resources.
plt.close()

# Close the SVS file to free-associated resources.
slide.close()
