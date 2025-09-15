import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl
import string
import ctypes
import os

tl.set_backend('numpy')

image = Image.open(os.path.dirname(__file__) + '/example_img.png').resize((128, 128))
image_np = np.array(image) / 255.0

core, factors = tucker(image_np, rank=[50, 50, 3])

reconstructed_tly = tucker_to_tensor((core, factors))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(np.clip(reconstructed_tly, 0, 1))
axes[1].set_title("Compressed (tensorly reconstruction)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

