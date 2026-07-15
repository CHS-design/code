import numpy as np
import torch
from PIL import Image
from os.path import splitext


def load_image(filename):
    """读取不同格式的图像或掩码文件，并统一返回 PIL 图像对象"""

    extension = splitext(filename)[1]

    if extension == '.npy':
        array = np.load(filename)
        return Image.fromarray(array)

    elif extension == '.pt':
        tensor = torch.load(filename)
        return Image.fromarray(tensor.numpy())

    elif extension == '.pth':
        tensor = torch.load(filename)
        return Image.fromarray(tensor.numpy())

    else:
        return Image.open(filename)
    
def unique_mask_values(index, mask_directory, mask_suffix):
    """读取一张掩码，找出其中出现过的所有类别值"""

    mask_files = list(mask_directory.glob(index + mask_suffix + '.*'))
    mask_file = mask_files[0]

    mask = np.asarray(load_image(mask_file))

    if mask.ndim == 2:
        return np.unique(mask)

    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)

    else:
        raise ValueError(
            f'掩码应当是二维或三维，实际维度为 {mask.ndim}'
        )
    