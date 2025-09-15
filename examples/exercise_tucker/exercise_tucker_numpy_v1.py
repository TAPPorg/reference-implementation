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
    ndim = core.ndim

    subscripts = string.ascii_lowercase

    core_subs = subscripts[:ndim]
    factor_subs = [f"{subscripts[ndim+i]}{core_subs[i]}" for i in range(ndim)]
    output_subs = ''.join([s[0] for s in factor_subs])

    einsum_str = f"{core_subs}," + ','.join(factor_subs) + f"->{output_subs}"

    result = np.einsum(einsum_str, core, *factors)
    return result

tl.set_backend('numpy')

image = Image.open(os.path.dirname(__file__) + '/example_img.png').resize((128, 128))
image_np = np.array(image) / 255.0

core, factors = tucker(image_np, rank=[50, 50, 3])

reconstructed = tucker_to_tensor_numpy(core, factors)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(np.clip(reconstructed, 0, 1))
axes[1].set_title("Compressed")
axes[1].axis('off')

plt.tight_layout()
plt.show()

