##########################################################################################
# This script reads an image and plots the distribution of pixel intensities in the image.
# The resulting values can then be used to "grab" the object in the image based on the
# pixel intensity distribution.
##########################################################################################

# Read image
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


datapath = Path(__file__).parent.parent / "data"

# Read image
img_bgr_full = cv2.imread((datapath / "orange2.png").as_posix())
img_hsv_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2HSV)

# Flatten the image
img_bgr = img_bgr_full.reshape(-1, 3)
img_hsv = img_hsv_full.reshape(-1, 3)

# Filter out black pixels
img_bgr = img_bgr[img_bgr.sum(axis=1) > 0]
img_hsv = img_hsv[img_hsv[:, 2] > 0]

# Split the channels
red = img_bgr[:, 2]
green = img_bgr[:, 1]
blue = img_bgr[:, 0]
hue = img_hsv[:, 0]
saturation = img_hsv[:, 1]
value = img_hsv[:, 2]

# Create a boxplot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(
    [red, green, blue, hue, saturation, value],
    labels=["Red", "Green", "Blue", "Hue", "Saturation", "Value"],
)
plt.title("Pixel Intensity Distribution")
plt.ylabel("Intensity")

# Get the whiskers
limits = np.array([item.get_ydata()[1] for item in boxplot["whiskers"]]).reshape(-1, 2)
limit_red, limit_green, limit_blue, limit_hue, limit_saturation, limit_value = limits
print(f"RGB limits: {limit_red}, {limit_green}, {limit_blue}")
print(f"HSV limits: {limit_hue}, {limit_saturation}, {limit_value}")

# plt.close()
plt.show()
