'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import the openslide library for reading whole-slide images.
import openslide
# Import matplotlib.pyplot for plotting images and figures.
import matplotlib.pyplot as plt

# Define the path to the SVS file used in this example.
svsFile = r"Data/A02.svs"

# Open the SVS file as an OpenSlide object for further operations.
slide = openslide.OpenSlide(svsFile)

# Retrieve the width and height of the slide at level 0.
slideWidth, slideHeight = slide.dimensions
# Retrieve the number of resolution levels available in the slide.
slideLevels = slide.level_count
# Retrieve the dimensions tuple for level 0 of the slide.
slideDimLevel0 = slide.level_dimensions[0]
# Retrieve the dimensions tuple for level 1 of the slide.
slideDimLevel1 = slide.level_dimensions[1]
# Retrieve the dimensions tuple for level 2 of the slide.
slideDimLevel2 = slide.level_dimensions[2]

# Print the main slide dimensions to the console for inspection.
print("Slide dimensions: ", slideWidth, slideHeight)
# Print the number of levels available in the slide to the console.
print("Slide levels: ", slideLevels)
# Print the dimensions of level 0 to the console for verification.
print("Slide dimensions level 0: ", slideDimLevel0)
# Print the dimensions of level 1 to the console for verification.
print("Slide dimensions level 1: ", slideDimLevel1)
# Print the dimensions of level 2 to the console for verification.
print("Slide dimensions level 2: ", slideDimLevel2)

# Print the downsample factors for each level relative to level 0.
print("Level downsamples:")
# Print each level's dimensions paired with its downsample factor.
for i, (dim, ds) in enumerate(zip(slide.level_dimensions, slide.level_downsamples)):
  print(f"  Level {i}: dimensions={dim}, downsample={round(ds, 4)}")

# Print how many metadata properties the slide exposes and list them.
print("-" * 50)
print(f"Slide properties count: {len(slide.properties)}")
# Show all available metadata keys and values for inspection.
for k in sorted(slide.properties.keys()):
  print(" ", k, "=", slide.properties[k])

# Create a new matplotlib figure to show the extracted region.
plt.figure()
# Display the extracted region from the slide in the current axes.
plt.imshow(
  # Read a rectangular region from the slide at the specified location, level, and size.
  slide.read_region(
    (20000, 20000),  # (x, y).
    2,  # Level.
    (1000, 1000)  # (width, height).
  )
)
# Add a title to the figure for context.
plt.title("Extracted Region from WSI at Level 2")
# Disable axis drawing to present a clean image without ticks or labels.
plt.axis("off")
# Adjust the layout to remove any extra whitespace around the figure.
plt.tight_layout()
# Render the figure window and block until it is closed by the user.
plt.show()
# Close the current figure and free its associated resources.
plt.close()
