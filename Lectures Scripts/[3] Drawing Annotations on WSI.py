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
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

svsFile = r"A02.svs"
xmlFile = r"A02.xml"

# Open the SVS file.
slide = openslide.OpenSlide(svsFile)
anList = []  # List of annotations.

# Read the XML file.
tree = ET.parse(xmlFile)
root = tree.getroot()
annotations = root.findall(".//Annotation")
for annotation in annotations:
  regions = annotation.findall(".//Region")
  for region in regions:
    vertices = region.findall(".//Vertex")
    coords = [
      (
        int(float(vertex.attrib["X"])),
        int(float(vertex.attrib["Y"]))
      )
      for vertex in vertices
    ]
    anList.append(
      {
        "Text"  : region.attrib["Text"],
        "Coords": coords
      }
    )

thumbnail = slide.get_thumbnail(
  (slide.dimensions[0] // 100, slide.dimensions[1] // 100)
)
plt.figure()
plt.imshow(thumbnail)
plt.axis("off")
plt.tight_layout()
for an in anList:
  x = [coord[0] // 100 for coord in an["Coords"]]
  y = [coord[1] // 100 for coord in an["Coords"]]
  if (an["Text"].strip() == "Benign"):
    color = "g"
  elif (an["Text"].strip() == "Carcinoma in situ"):
    color = "y"
  elif (an["Text"].strip() == "Carcinoma invasive"):
    color = "r"
  else:
    color = "b"  # Normal
  # Create a polygon.
  plt.fill(x, y, color, alpha=0.5)
  plt.plot(x + [x[0]], y + [y[0]], color)
  plt.text(
    min(x),
    min(y),
    an["Text"],
    fontsize=5,
    color="black",
  )
plt.show()

# Close the SVS file.
slide.close()
