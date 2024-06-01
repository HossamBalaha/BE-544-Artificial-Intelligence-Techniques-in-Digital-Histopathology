# Author: Hossam Magdy Balaha
# Date: May 30th, 2024

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
import numpy as np

svsFile = r"A02.svs"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)

# Find the properties of the slide.
slideWidth, slideHeight = slide.dimensions
slideLevels = slide.level_count

(x, y) = (19000, 25000)
(width, height) = (512, 512)

plt.figure()
for i in range(slideLevels):
  # Calculate the downsample ratio with reference to the highest resolution.
  dRatio = int(slide.level_downsamples[i] / slide.level_downsamples[0])

  # Calculate the downsample factor.
  factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
  factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

  xNew = x - dRatio * factorWidth
  yNew = y - dRatio * factorHeight

  regionSlide = slide.read_region(
    (xNew, yNew),
    i,
    (width, height),
  )
  regionSlide = regionSlide.convert("RGB")
  regionSlide = np.array(regionSlide)

  plt.subplot(2, slideLevels, i + 1)
  plt.imshow(regionSlide)
  plt.axis("off")
  plt.tight_layout()
  plt.title(f"Level {i} ({dRatio}x)")

  verify = regionSlide[
           factorHeight:factorHeight + height // dRatio,
           factorWidth:factorWidth + width // dRatio,
           ]

  plt.subplot(2, slideLevels, 3 + i + 1)
  plt.imshow(verify)
  plt.axis("off")
  plt.tight_layout()
  plt.title(f"Cropped to Verify Level {i}")
plt.show()
