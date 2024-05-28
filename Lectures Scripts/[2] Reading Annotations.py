import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

xmlFile = r"A02.xml"

anList = []  # List of annotations.

# Read the XML file.
tree = ET.parse(xmlFile)
root = tree.getroot()


annotations = root.findall(".//Annotation")
print("Number of annotations: ", len(annotations))


for annotation in annotations:
  regions = annotation.findall(".//Region")
  print("- Number of regions: ", len(regions))
  for region in regions:
    print("-- Region Text: ", region.attrib["Text"])
    vertices = region.findall(".//Vertex")
    print("-- Number of vertices: ", len(vertices))

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


print("Number of annotations: ", len(anList))
