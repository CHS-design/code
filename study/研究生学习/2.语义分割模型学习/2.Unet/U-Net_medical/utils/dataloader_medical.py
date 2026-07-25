import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, data_path, input_shape, num_classes, augmentation=True, txt_name: str = "train.txt"):
        super(UnetDataset, self).__init__()

    # CAMUS 的划分文件位于：
    # C:\Users\admin\Desktop\CAMUS\splits\train.txt
    # C:\Users\admin\Desktop\CAMUS\splits\val.txt
        with open(os.path.join(data_path, "splits", txt_name), "r", encoding="ascii") as f:
    # 每一行只有一个样本 ID，例如：163
            self.annotation_lines = f.readlines()

        # 初始化其他参数
        self.length = len(self.annotation_lines)  # 数据集的长度
        self.input_shape = input_shape  # 输入图像的形状（宽和高）
        self.num_classes = num_classes  # 类别数目
        self.augmentation = augmentation  # 是否在训练阶段，用来控制是否使用数据增强
        self.data_path = data_path  # 数据集路径

    def __len__(self):
        # 返回数据集的大小
        return self.length

    def __getitem__(self, index):
        # 从 train.txt 或 val.txt 取出一个样本 ID，例如 "163\n"。
        annotation_line = self.annotation_lines[index]
        name = annotation_line.strip()

        # CAMUS 中图像与掩码文件名一一对应：
        # train_images/163.png <-> train_masks/163.png
        jpg = Image.open(
            os.path.join(self.data_path, "train_images", name + ".png")
        )

        # 掩码必须是单通道 L 模式。
        # 这样标签会保持为 0、85、170、255，而不会带有额外颜色通道。
        png = Image.open(
            os.path.join(self.data_path, "train_masks", name + ".png")
        ).convert("L")

        # 图像与掩码必须执行完全相同的几何变换。
        # 图像使用双三次插值，掩码在 get_random_data 中使用最近邻插值，
        # 因而不会产生不属于任何类别的中间标签。
        jpg, png = self.get_random_data(
            jpg, png, self.input_shape, random=self.augmentation
        )

        # cvtColor 会把灰度图复制为 3 通道 RGB，
        # 因此仍能输入 ResNet-50 的 3 通道第一层卷积。
        # 转置后形状为 (3, H, W)，并将像素从 0~255 缩放到 0~1。
        jpg = np.transpose(
            preprocess_input(np.array(jpg, np.float64)),
            [2, 0, 1]
        )

        # 读取增强后的单通道掩码，形状为 (H, W)。
        raw_mask = np.array(png, dtype=np.uint8)

        # CAMUS 只允许这四个原始像素值。
        # 先检查能避免异常掩码被悄悄当作某一类参与训练。
        raw_values = np.unique(raw_mask)
        valid_values = np.array([0, 85, 170, 255], dtype=np.uint8)
        if not np.isin(raw_values, valid_values).all():
            raise ValueError(f"Unexpected mask values in {name}: {raw_values}")

        # 将 CAMUS 原始掩码值映射成 CrossEntropyLoss 需要的连续类别 ID：
        # 0   -> 背景
        # 85  -> 左心室腔
        # 170 -> 心肌
        # 255 -> 左心房
        modify_png = np.zeros_like(raw_mask, dtype=np.int64)
        modify_png[raw_mask == 85] = 1
        modify_png[raw_mask == 170] = 2
        modify_png[raw_mask == 255] = 3

        # modify_png 供交叉熵损失使用，形状为 (H, W)，值只能是 0~3。
        #
        # seg_labels 供项目现有 Dice_loss 使用。
        # 该 Dice_loss 约定标签最后多保留一个 ignore 通道，
        # 所以当第 3 步把 num_classes 统一为 4 时，
        # seg_labels 形状会是 (H, W, 5)，最后一维不是第五种解剖类别。
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[
            modify_png.reshape([-1])
        ]
        seg_labels = seg_labels.reshape(
            int(self.input_shape[0]),
            int(self.input_shape[1]),
            self.num_classes + 1,
        )

        return jpg, modify_png, seg_labels

    # 生成随机数的函数
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 对图像和标签进行随机数据增强的函数
    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 将图像转为RGB格式
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size  # 获取图像的宽和高
        h, w = input_shape  # 获取目标图像的高和宽

        if not random:
            # 如果不进行随机增强（例如在验证阶段）
            iw, ih = image.size
            scale = min(w / iw, h / ih)  # 计算缩放比例
            nw = int(iw * scale)  # 根据比例计算缩放后的宽
            nh = int(ih * scale)  # 根据比例计算缩放后的高

            # 缩放图像并进行中心裁剪
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))  # 创建一个灰色背景的图像
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将缩放后的图像粘贴到目标图像中

            # 缩放标签图像并进行中心裁剪
            label = label.resize((nw, nh), Image.NEAREST)  # 标签使用最近邻插值
            new_label = Image.new('L', [w, h], (0))  # 创建一个空白标签图像
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))  # 将缩放后的标签图像粘贴到目标图像中
            return new_image, new_label

        # 获取一个新的宽高比（通过调整宽和高的比例）
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)  # 随机缩放比例
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 随机翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 在图像周围随机添加灰色边框
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))  # 将图像粘贴到新图像中
        new_label.paste(label, (dx, dy))  # 将标签粘贴到新标签中
        image = new_image
        label = new_label

        # 转换图像为数组
        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1  # 随机调整色调、饱和度和亮度
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))  # 转为HSV色域
        dtype = image_data.dtype  # 获取数据类型
        x = np.arange(0, 256, dtype=r.dtype)  # 获取颜色值范围
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 应用色调变换
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 应用饱和度变换
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 应用亮度变换

        # 使用查找表（LUT）应用变换
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)  # 转换回RGB色域

        return image_data, label  # 返回经过增强的图像和标签


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    # 初始化三个列表，用于存储每个批次中的图像、标签和one-hot编码标签
    images = []  # 用来存储图像数据
    pngs = []  # 用来存储原始标签（通常是类别标签）
    seg_labels = []  # 用来存储one-hot编码的标签

    # 遍历当前批次中的每个样本（img, png, labels）
    for img, png, labels in batch:
        images.append(img)    # 将图像添加到images列表中
        pngs.append(png)       # 将原始标签添加到pngs列表中
        seg_labels.append(labels)   # 将one-hot标签添加到seg_labels列表中

    # 将列表转换为NumPy数组，然后转换为torch张量
    # images的张量需要是float类型，通常用于输入图像
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # pngs的张量需要是long类型，通常用于标签索引
    pngs = torch.from_numpy(np.array(pngs)).long()
    # seg_labels的张量需要是float类型，通常用于标签的one-hot编码
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)

    # 返回三个张量，分别对应图像、原始标签和one-hot标签
    return images, pngs, seg_labels