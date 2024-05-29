# Author: Hossam Magdy Balaha
# Date: May 28th, 2024

# -------------------------------------------------- #
# https://openslide.org/api/python/#installing
OPENSLIDE_PATH = r"C:\openslide-bin-4.0.0.3-windows-x64\bin"

import os

if (hasattr(os, "add_dll_directory")):
  with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
else:
  import openslide
# -------------------------------------------------- #

import openslide
import matplotlib.pyplot as plt

svsFile = r"A02.svs"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Find the properties of the slide.
slideWidth, slideHeight = slide.dimensions
slideLevels = slide.level_count
slideDimLevel0 = slide.level_dimensions[0]
slideDimLevel1 = slide.level_dimensions[1]
slideDimLevel2 = slide.level_dimensions[2]

print("Slide dimensions: ", slideWidth, slideHeight)
print("Slide levels: ", slideLevels)
print("Slide dimensions level 0: ", slideDimLevel0)
print("Slide dimensions level 1: ", slideDimLevel1)
print("Slide dimensions level 2: ", slideDimLevel2)

# Display a portion of the slide.
plt.figure()
plt.imshow(
  slide.read_region(
    (20000, 20000),  # (x, y)
    2,  # level
    (1000, 1000)  # (width, height)
  )
)
plt.axis("off")
plt.tight_layout()
plt.show()
