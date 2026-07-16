import torch
import torch.nn.functional as F
import os
import logging
from utils.data_loading import BasicDataset
import argparse
import numpy as np
from PIL import Image
from unet import UNet

def predict_img(
    network,
    full_image,
    device,
    scale_factor=1.0,
    output_threshold=0.5
):
    """对单张图像进行预测，并返回像素类别掩码"""

    network.eval()

    preprocessed_image = BasicDataset.preprocess(
        mask_values=None,
        pil_image=full_image,
        scale=scale_factor,
        is_mask=False
    )

    image_tensor = torch.from_numpy(
        preprocessed_image
    )

    image_tensor = image_tensor.unsqueeze(dim=0)

    image_tensor = image_tensor.to(
        device=device,
        dtype=torch.float32
    )

    with torch.no_grad():
        output_logits = network(image_tensor)

        original_height = full_image.size[1]
        original_width = full_image.size[0]

        output_logits = F.interpolate(
            output_logits,
            size=(original_height, original_width),
            mode='bilinear'
        )

        if network.n_classes > 1:
            predicted_mask = output_logits.argmax(dim=1)

        else:
            output_probabilities = torch.sigmoid(
                output_logits
            )

            predicted_mask = (
                output_probabilities > output_threshold
            )

    predicted_mask = (
        predicted_mask[0]
        .long()
        .squeeze()
        .cpu()
        .numpy()
    )

    return predicted_mask
def get_args():
    """解析预测程序的命令行参数"""

    parser = argparse.ArgumentParser(
        description='使用 U-Net 预测图像掩码'
    )

    parser.add_argument(
        '--model',
        '-m',
        dest='model_path',
        type=str,
        default='MODEL.pth',
        metavar='FILE',
        help='模型权重文件路径'
    )

    parser.add_argument(
        '--input',
        '-i',
        dest='input_files',
        nargs='+',
        required=True,
        metavar='INPUT',
        help='需要预测的图像文件路径'
    )

    parser.add_argument(
        '--output',
        '-o',
        dest='output_files',
        nargs='+',
        metavar='OUTPUT',
        help='预测结果的保存路径'
    )

    parser.add_argument(
        '--viz',
        '-v',
        action='store_true',
        default=False,
        help='显示预测结果'
    )

    parser.add_argument(
        '--no-save',
        '-n',
        dest='do_not_save',
        action='store_true',
        default=False,
        help='不保存预测掩码'
    )

    parser.add_argument(
        '--mask-threshold',
        '-t',
        dest='mask_threshold',
        type=float,
        default=0.5,
        help='二分类中判断前景的概率阈值'
    )

    parser.add_argument(
        '--scale',
        '-s',
        dest='scale_factor',
        type=float,
        default=0.5,
        help='输入图像缩放比例'
    )

    parser.add_argument(
        '--bilinear',
        action='store_true',
        default=False,
        help='使用双线性插值进行上采样'
    )

    parser.add_argument(
        '--classes',
        '-c',
        dest='number_of_classes',
        type=int,
        default=2,
        help='分割类别数量'
    )

    return parser.parse_args()
def get_output_filenames(args):
    """生成每张输入图像对应的输出掩码路径"""

    if args.output_files is not None:
        return args.output_files

    output_files = []

    for input_file in args.input_files:
        file_name_without_extension, file_extension = os.path.splitext(
            input_file
        )

        output_file = (
            file_name_without_extension
            + '_OUT.png'
        )

        output_files.append(output_file)

    return output_files
def mask_to_image(mask: np.ndarray, mask_values):
    """将类别索引掩码转换为可以保存的 PIL 图像"""

    if isinstance(mask_values[0], list):
        output_image = np.zeros(
            (
                mask.shape[-2],
                mask.shape[-1],
                len(mask_values[0])
            ),
            dtype=np.uint8
        )

    elif mask_values == [0, 1]:
        output_image = np.zeros(
            (
                mask.shape[-2],
                mask.shape[-1]
            ),
            dtype=bool
        )

    else:
        output_image = np.zeros(
            (
                mask.shape[-2],
                mask.shape[-1]
            ),
            dtype=np.uint8
        )

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for class_index, mask_value in enumerate(mask_values):
        output_image[mask == class_index] = mask_value

    return Image.fromarray(output_image)


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    input_files = args.input_files
    output_files = get_output_filenames(args)

    if args.output_files is not None:
        if len(output_files) != len(input_files):
            raise ValueError(
                '输出文件数量必须与输入文件数量相同'
            )

    network = UNet(
        n_channels=3,
        n_classes=args.number_of_classes,
        bilinear=args.bilinear
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logging.info(
        f'正在加载模型：{args.model_path}'
    )

    logging.info(
        f'使用计算设备：{device}'
    )

    network.to(device=device)

    state_dict = torch.load(
        args.model_path,
        map_location=device
    )

    mask_values = state_dict.pop(
        'mask_values',
        [0, 1]
    )

    network.load_state_dict(state_dict)

    logging.info('模型加载完成')

    for index, input_file in enumerate(input_files):
        logging.info(
            f'正在预测图像：{input_file}'
        )

        image = Image.open(input_file)

        predicted_mask = predict_img(
            network=network,
            full_image=image,
            scale_factor=args.scale_factor,
            output_threshold=args.mask_threshold,
            device=device
        )

        if not args.do_not_save:
            output_file = output_files[index]

            result_image = mask_to_image(
                predicted_mask,
                mask_values
            )

            result_image.save(output_file)

            logging.info(
                f'预测掩码已保存到：{output_file}'
            )

        if args.viz:
            from utils.utils import plot_img_and_mask

            logging.info(
                f'正在显示图像 {input_file} 的预测结果'
            )

            plot_img_and_mask(
                image,
                predicted_mask
            )
