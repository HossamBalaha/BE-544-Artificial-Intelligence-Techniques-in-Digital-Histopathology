'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import the OpenSlide Python wrapper and OpenCV for image operations.
import openslide, cv2
# Import matplotlib.pyplot for plotting results.
import matplotlib.pyplot as plt
# Import the helper function to extract BACH annotations from XML.
from HMB_Spring_2026_Helpers import ExtractBACHAnnotationsFromXML, ExtractWSIRegion

# Define the path to the SVS file to open.
svsFile = r"Data/A02.svs"
# Define the path to the XML file that contains annotations.
xmlFile = r"Data/A02.xml"

# Open the SVS file as an OpenSlide object for reading image regions.
slide = openslide.OpenSlide(svsFile)

# Extract the list of annotation dictionaries from the XML file.
anList = ExtractBACHAnnotationsFromXML(xmlFile, verbose=False)

# Select the first parsed annotation to extract a region from the whole-slide image.
region = anList[0]
# Use the helper function to extract the region image, mask, and ROI from the slide.
regionImage, regionMask, roi = ExtractWSIRegion(slide, region)

# Create a new matplotlib figure for displaying the mask, image, and ROI.
plt.figure()
# Create the first subplot to show the binary mask image.
plt.subplot(1, 3, 1)
# Display the region mask using a gray colormap.
plt.imshow(regionMask, cmap="gray")
# Disable the axis on the mask subplot.
plt.axis("off")
# Set the title for the mask subplot.
plt.title("Region Mask")
# Create the second subplot to show the region image.
plt.subplot(1, 3, 2)
# Display the extracted region image.
plt.imshow(regionImage)
# Disable the axis on the region image subplot.
plt.axis("off")
# Set the title for the region image subplot.
plt.title("Region Image")
# Create the third subplot to show the masked ROI.
plt.subplot(1, 3, 3)
# Display the ROI image after masking.
plt.imshow(roi)
# Disable the axis on the ROI subplot.
plt.axis("off")
# Set the title for the ROI subplot.
plt.title("Region ROI")
# Render the figure window and block until it is closed by the user.
plt.show()
# Close the current figure and free its associated resources.
plt.close()

# Close the SVS file to free-associated resources.
slide.close()
