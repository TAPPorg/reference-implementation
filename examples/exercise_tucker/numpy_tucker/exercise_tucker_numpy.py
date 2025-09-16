import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl
import string
import ctypes
import os

def tucker_to_tensor_numpy(core, factors):
    nmode = core.ndim
    indices = string.ascii_lowercase

    # Create subscripts
    idx_core = indices[:nmode]
    idx_factor = [f"{indices[nmode+i]}{idx_core[i]}" for i in range(nmode)]
    idx_output = [''.join([s for s in idx_core + idx_factor[0] if s not in ''.join(set(idx_core) & set(idx_factor[0]))])]
    for i in range(1, nmode):
        idx_output = idx_output + [''.join([s for s in idx_output[i - 1] + idx_factor[i] if s not in ''.join(set(idx_factor[i]) & set(idx_output[i - 1]))])]

    # Contracting first factor
    einsum_str = f"{idx_core}," + idx_factor[0] + f"->{idx_output[0]}"
    result = np.einsum(einsum_str, core, factors[0])
    
    # Further contracting factors
    for i in range(1, nmode):
        einsum_str = f"{idx_output[i - 1]}," + idx_factor[i] + f"->{idx_output[i]}"
        result = np.einsum(einsum_str, result, factors[i])

    return result

# Load image
image = Image.open(os.path.dirname(__file__) + '/example_img.png').resize((128, 128))
# Format data
image_np = np.array(image) / 255.0

# Compress image
core, factors = tucker(image_np, rank=[50, 50, 3])

# Reconstruct image
reconstructed = tucker_to_tensor_numpy(core, factors)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display original image
axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis('off')

# Display reconstructed
axes[1].imshow(np.clip(reconstructed, 0, 1))
axes[1].set_title("Compressed")
axes[1].axis('off')

plt.tight_layout()
plt.show()

