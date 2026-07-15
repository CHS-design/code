import numpy as np
import torch
from PIL import Image
from os.path import splitext
import logging
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm

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
    
class BasicDataset(Dataset):
    """读取图像与对应掩码的基础数据集"""

    def __init__(
        self,
        images_directory,
        masks_directory,
        scale=1.0,
        mask_suffix=''
    ):
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)

        if scale <= 0:
            raise AssertionError('缩放比例必须大于 0')

        if scale > 1:
            raise AssertionError('缩放比例必须小于或等于 1')

        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = []

        for file_name in listdir(self.images_directory):
            file_path = join(self.images_directory, file_name)
            is_regular_file = isfile(file_path)
            is_hidden_file = file_name.startswith('.')

            if is_regular_file and not is_hidden_file:
                image_id = splitext(file_name)[0]
                self.ids.append(image_id)

        if len(self.ids) == 0:
            raise RuntimeError(
                f'在 {self.images_directory} 中没有找到输入图像'
            )

        logging.info(f'数据集包含 {len(self.ids)} 个样本')
        logging.info('正在扫描所有掩码中的类别值')

        with Pool() as process_pool:
            unique_values_by_mask = list(
                tqdm(
                    process_pool.imap(
                        partial(
                            unique_mask_values,
                            mask_directory=self.masks_directory,
                            mask_suffix=self.mask_suffix
                        ),
                        self.ids
                    ),
                    total=len(self.ids)
                )
            )

        all_mask_values = np.concatenate(unique_values_by_mask, axis=0)
        unique_mask_values_in_dataset = np.unique(
            all_mask_values,
            axis=0
        )

        self.mask_values = list(
            sorted(unique_mask_values_in_dataset.tolist())
        )

        logging.info(f'掩码类别值为: {self.mask_values}')

    def __len__(self):
        """返回数据集中的样本数量"""

        return len(self.ids)
    
    @staticmethod
    def preprocess(mask_values, pil_image, scale, is_mask):
        """缩放图像或掩码，并转换为后续可用的 NumPy 数组"""

        width, height = pil_image.size

        new_width = int(scale * width)
        new_height = int(scale * height)

        if new_width <= 0:
            raise AssertionError('缩放比例过小，图像宽度变为 0')

        if new_height <= 0:
            raise AssertionError('缩放比例过小，图像高度变为 0')

        if is_mask:
            resampling_method = Image.NEAREST
        else:
            resampling_method = Image.BICUBIC

        pil_image = pil_image.resize(
            (new_width, new_height),
            resample=resampling_method
        )

        image_array = np.asarray(pil_image)

        if is_mask:
            mask = np.zeros(
                (new_height, new_width),
                dtype=np.int64
            )

            for class_index, mask_value in enumerate(mask_values):
                if image_array.ndim == 2:
                    mask[image_array == mask_value] = class_index

                else:
                    same_color = np.all(
                        image_array == mask_value,
                        axis=-1
                    )
                    mask[same_color] = class_index

            return mask

        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=0)

        else:
            image_array = image_array.transpose((2, 0, 1))

        if np.any(image_array > 1):
            image_array = image_array / 255.0

        return image_array 

    def __getitem__(self, index):
        """读取并返回指定索引处的一组图像和掩码"""

        image_id = self.ids[index]

        mask_files = list(
            self.masks_directory.glob(
                image_id + self.mask_suffix + '.*'
            )
        )

        image_files = list(
            self.images_directory.glob(image_id + '.*')
        )

        if len(image_files) != 1:
            raise AssertionError(
                f'图像编号 {image_id} 对应的原图数量应为 1，'
                f'实际找到: {image_files}'
            )

        if len(mask_files) != 1:
            raise AssertionError(
                f'图像编号 {image_id} 对应的掩码数量应为 1，'
                f'实际找到: {mask_files}'
            )

        mask = load_image(mask_files[0])
        image = load_image(image_files[0])

        if image.size != mask.size:
            raise AssertionError(
                f'图像与掩码尺寸不一致：'
                f'图像为 {image.size}，掩码为 {mask.size}'
            )

        image = self.preprocess(
            self.mask_values,
            image,
            self.scale,
            is_mask=False
        )

        mask = self.preprocess(
            self.mask_values,
            mask,
            self.scale,
            is_mask=True
        )

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        } 



class CarvanaDataset(BasicDataset):
    """适配 Carvana 数据集文件命名规则的数据集"""

    def __init__(self, images_directory, masks_directory, scale=1.0):
        super().__init__(
            images_directory,
            masks_directory,
            scale,
            mask_suffix='_mask'
        )
    