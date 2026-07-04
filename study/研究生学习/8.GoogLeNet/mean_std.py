from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data_me' / 'train'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
IMAGE_SIZE = (224, 224)

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR


def iter_image_paths(data_dir):
    for image_path in data_dir.rglob('*'):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            yield image_path


def load_image_array(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB').resize(IMAGE_SIZE, RESAMPLE)
        return np.asarray(image, dtype=np.float32) / 255.0


def main():
    image_paths = list(iter_image_paths(DATA_DIR))
    if not image_paths:
        raise FileNotFoundError(f'没有在 {DATA_DIR} 找到图片')

    pixel_count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)

    for image_path in image_paths:
        image_array = load_image_array(image_path)
        pixels = image_array.shape[0] * image_array.shape[1]
        pixel_count += pixels
        channel_sum += image_array.sum(axis=(0, 1))
        channel_squared_sum += (image_array ** 2).sum(axis=(0, 1))

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_squared_sum / pixel_count - mean ** 2)

    print('Image count:', len(image_paths))
    print('Mean:', mean.tolist())
    print('Std:', std.tolist())


if __name__ == '__main__':
    main()
