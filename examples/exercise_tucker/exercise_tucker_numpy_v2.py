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
    output_subs = [''.join([s for s in core_subs + factor_subs[0] if s not in ''.join(set(core_subs) & set(factor_subs[0]))])]
    for i in range(1, ndim):
        output_subs = output_subs + [''.join([s for s in output_subs[i - 1] + factor_subs[i] if s not in ''.join(set(factor_subs[i]) & set(output_subs[i - 1]))])]

    einsum_str = f"{core_subs}," + factor_subs[0] + f"->{output_subs[0]}"
    result = np.einsum(einsum_str, core, factors[0])
    
    for i in range(1, ndim):
        einsum_str = f"{output_subs[i - 1]}," + factor_subs[i] + f"->{output_subs[i]}"
        result = np.einsum(einsum_str, result, factors[i])

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

