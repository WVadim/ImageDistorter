import rawpy
import numpy as np

from typing import Any
import matplotlib.pyplot as plt
from image_distorter import ImageDistorter
import cv2


def load_image(path: str) -> Any:
    # Open the .dng file using rawpy
    with rawpy.imread(path) as raw:
        # Get the image data in the RGB color space
        rgb = raw.postprocess()
    return rgb


def save_image(image: Any, path: str):
    # Convert the image to the BGR color space
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Save the image to the specified path
    cv2.imwrite(path, bgr)


distorter = ImageDistorter(number_of_effects_range=(1, 1))

original_rgb = load_image("a0001-jmac_DSC1459.dng")

# distorted_rgb, text = distorter(original_rgb)
distorted_rgb, text, original_rgb = distorter(
    original_rgb
)
# print(", ".join(text))
distorted_rgb = np.clip(distorted_rgb, 0, 255).astype(np.uint8)
original_rgb = np.clip(original_rgb, 0, 255).astype(np.uint8)
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle(" ".join(text))

# Show the first image in the first subplot
ax1.imshow(distorted_rgb)

# Show the second image in the second subplot
ax2.imshow(original_rgb)

# Remove the x and y ticks from both subplots
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

# Show the figure
plt.show()
