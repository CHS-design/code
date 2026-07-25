# 导入标准库和第三方库
import torch  # 导入PyTorch库
from torch.utils.data import DataLoader  # 导入数据加载器模块

# 导入自定义模块和模型
from model.unet_resnet import Unet  # 导入U-Net模型
from utils.dataloader_medical import UnetDataset, unet_dataset_collate  # 导入U-Net数据集及其合并函数
from utils.train_and_eval import evaluate


def val(args):
    # num_classes 是总类别数：背景、左心室腔、心肌、左心房。
    num_classes = args.num_classes

    # 与训练脚本保持相同的设备选择逻辑。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # CAMUS 原始图像为 256 x 256，且 256 能被 32 整除。
    input_shape = [256, 256]

    # 只读取独立验证集，且禁止随机增强，保证指标可重复。
    val_dataset = UnetDataset(
        args.data_path,
        input_shape,
        num_classes,
        augmentation=False,
        txt_name="val.txt",
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=unet_dataset_collate,
    )

    # 输出为四个类别的得分图，形状为 (B, 4, 256, 256)。
    model = Unet(num_classes=num_classes)
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)

    # 复用训练阶段的验证逻辑，计算完整验证集的混淆矩阵和前景 mIoU。
    evaluate(
        model,
        val_loader,
        device,
        dice_loss=True,
        focal_loss=False,
        num_classes=num_classes,
    )



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="CAMUS U-Net validation")

    parser.add_argument(
        "--data-path",
        default=r"C:\Users\admin\Desktop\CAMUS",
        help="CAMUS 数据集根目录",
    )
    # 显式指定新训练的四分类权重，避免误用旧的二分类 checkpoint。
    parser.add_argument("--weights", required=True, help="best_model_4.pth 的路径")
    parser.add_argument(
        "--num-classes",
        default=4,
        type=int,
        help="总类别数，包含背景；CAMUS 固定为 4",
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", default="cuda", help="training device")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    val(args)
