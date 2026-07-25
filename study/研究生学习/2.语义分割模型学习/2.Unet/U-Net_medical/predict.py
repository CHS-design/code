import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unet_resnet import Unet
from utils.create_exp_folder import create_val_exp_folder
from utils.utils import cvtColor, preprocess_input, resize_image


# CAMUS 类别 ID 对应的 RGB 可视化颜色：
# 0=背景，1=左心室腔，2=心肌，3=左心房。
CAMUS_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 38, 38],
        [22, 163, 74],
        [250, 204, 21],
    ],
    dtype=np.uint8,
)


def time_synchronized(device):
    # 仅在实际使用 CUDA 时同步，CPU 推理不需要调用 CUDA API。
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time()


def load_model(model_path, num_classes, device):
    # 四分类模型输出形状为 (B, 4, 256, 256)。
    model = Unet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def detect_image(file_path, model, device, output_dir, overlay=True):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, OSError) as error:
        print(f"Error opening image: {error}")
        return

    # CAMUS 灰度图复制为 RGB 三通道，以匹配 ResNet-50 的输入层。
    image = cvtColor(image)
    original_image = image.copy()

    # 推理尺寸必须与训练尺寸相同。
    input_shape = [256, 256]
    original_h, original_w = np.array(image).shape[:2]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))

    # (H, W, 3) -> (1, 3, H, W)，并将像素值归一化到 0~1。
    image_data = np.expand_dims(
        np.transpose(
            preprocess_input(np.array(image_data, np.float32)),
            (2, 0, 1),
        ),
        axis=0,
    )

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)

        # 先缩放每类概率图，再取 argmax；不能直接缩放类别 ID。
        probabilities = F.softmax(model(images), dim=1)[0]
        probabilities = probabilities.permute(1, 2, 0).cpu().numpy()
        probabilities = probabilities[
            int((input_shape[0] - nh) // 2):int((input_shape[0] - nh) // 2 + nh),
            int((input_shape[1] - nw) // 2):int((input_shape[1] - nw) // 2 + nw),
        ]
        probabilities = cv2.resize(
            probabilities,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction = probabilities.argmax(axis=-1).astype(np.uint8)

    stem = Path(file_path).stem

    # 原始掩码像素值严格为 0/1/2/3，可供后续定量分析使用。
    label_path = os.path.join(output_dir, f"{stem}_label.png")
    Image.fromarray(prediction, mode="L").save(label_path)

    color_mask = CAMUS_COLORS[prediction]
    if overlay:
        # 将预测颜色叠加到原始超声图像，便于人工检查。
        result = cv2.addWeighted(np.array(original_image), 0.45, color_mask, 0.55, 0)
        output_path = os.path.join(output_dir, f"{stem}_overlay.png")
    else:
        result = color_mask
        output_path = os.path.join(output_dir, f"{stem}_color.png")

    Image.fromarray(result).save(output_path)
    print(f"Saved label: {label_path}")
    print(f"Saved visualization: {output_path}")


def predict(args):
    output_dir = create_val_exp_folder()
    num_classes = args.num_classes

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, num_classes, device)

    if os.path.isdir(args.data_path):
        file_paths = sorted(
            str(path)
            for path in Path(args.data_path).rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    elif os.path.isfile(args.data_path):
        file_paths = [args.data_path]
    else:
        raise ValueError(f"Unsupported input path: {args.data_path}")

    if not file_paths:
        raise ValueError(f"No supported images found in: {args.data_path}")

    start_time = time_synchronized(device)
    for file_path in file_paths:
        detect_image(file_path, model, device, output_dir, overlay=args.overlay)
    end_time = time_synchronized(device)

    print(f"inference time for: {end_time - start_time:.2f}s")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="CAMUS U-Net prediction")
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        default=r"C:\Users\admin\Desktop\CAMUS\test_images",
        help="待预测的一张图像或图像目录",
    )
    parser.add_argument("--weights", required=True, help="best_model_4.pth 的路径")
    parser.add_argument("--num-classes", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="仅保存彩色类别图，不保存原图叠加效果",
    )
    parser.set_defaults(overlay=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)
