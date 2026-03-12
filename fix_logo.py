import numpy as np
from PIL import Image

# Load the logo
img = Image.open("assets/logo.png").convert("RGBA")
data = np.array(img)

# Separate color channels
r, g, b, a = data.T

# Detect black pixels
black = (r < 15) & (g < 15) & (b < 15)

# Make them transparent
data[..., -1][black.T] = 0

# Save new image
Image.fromarray(data).save("assets/logo_transparent.png")

print("Transparent logo created: assets/logo_transparent.png")
