import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


images_directory = Path('./data/imgs/')
masks_directory = Path('./data/masks/')
checkpoints_directory = Path('./checkpoints/')
def train_model(
    model,
    device,
    epochs=5,
    batch_size=1,
    learning_rate=1e-5,
    validation_percentage=0.1,
    save_checkpoint=True,
    image_scale=0.5,
    amp=False,
    weight_decay=1e-8,
    momentum=0.999,
    gradient_clipping=1.0
):
    """完成数据准备、模型训练、验证与权重保存"""

    try:
        dataset = CarvanaDataset(
            images_directory,
            masks_directory,
            image_scale
        )

    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(
            images_directory,
            masks_directory,
            image_scale
        )

    number_of_validation_samples = int(
        len(dataset) * validation_percentage
    )

    number_of_training_samples = (
        len(dataset) - number_of_validation_samples
    )

    split_generator = torch.Generator()
    split_generator.manual_seed(0)

    training_dataset, validation_dataset = random_split(
        dataset,
        [
            number_of_training_samples,
            number_of_validation_samples
        ],
        generator=split_generator
    )

    number_of_workers = os.cpu_count()

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_workers,
        pin_memory=True
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=True,
        drop_last=True
    )


    experiment = wandb.init(
        project='U-Net',
        resume='allow',
        anonymous='must'
    )

    experiment.config.update(
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_percentage': validation_percentage,
            'save_checkpoint': save_checkpoint,
            'image_scale': image_scale,
            'amp': amp
        }
    )

    logging.info(
        f'开始训练：\n'
        f'    训练轮数：{epochs}\n'
        f'    批次大小：{batch_size}\n'
        f'    学习率：{learning_rate}\n'
        f'    训练集样本数：{number_of_training_samples}\n'
        f'    验证集样本数：{number_of_validation_samples}\n'
        f'    保存权重：{save_checkpoint}\n'
        f'    计算设备：{device.type}\n'
        f'    图像缩放比例：{image_scale}\n'
        f'    自动混合精度：{amp}'
    )

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True
    )

    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5
    )

    gradient_scaler = torch.cuda.amp.GradScaler(
        enabled=amp
    )

    if model.n_classes > 1:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()

    global_step = 0

    if device.type == 'mps':
        autocast_device_type = 'cpu'
    else:
        autocast_device_type = device.type

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0

        with tqdm(
            total=number_of_training_samples,
            desc=f'第 {epoch}/{epochs} 轮训练',
            unit='图像'
        ) as progress_bar:

            for batch in training_data_loader:
                images = batch['image']
                target_masks = batch['mask']

                input_channel_count = images.shape[1]

                if input_channel_count != model.n_channels:
                    raise AssertionError(
                        f'网络期望输入通道数为 {model.n_channels}，'
                        f'实际读入通道数为 {input_channel_count}。'
                        f'请检查图像读取与预处理流程。'
                    )

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last
                )

                target_masks = target_masks.to(
                    device=device,
                    dtype=torch.long
                )

                with torch.autocast(
                    device_type=autocast_device_type,
                    enabled=amp
                ):
                    predicted_logits = model(images)

                    if model.n_classes == 1:
                        predicted_logits_without_channel = (
                            predicted_logits.squeeze(dim=1)
                        )

                        target_masks_as_float = target_masks.float()

                        pixel_classification_loss = loss_function(
                            predicted_logits_without_channel,
                            target_masks_as_float
                        )

                        dice_component_loss = dice_loss(
                            torch.sigmoid(
                                predicted_logits_without_channel
                            ),
                            target_masks_as_float,
                            multiclass=False
                        )

                        loss = (
                            pixel_classification_loss
                            + dice_component_loss
                        )

                    else:
                        pixel_classification_loss = loss_function(
                            predicted_logits,
                            target_masks
                        )

                        predicted_probabilities = F.softmax(
                            predicted_logits,
                            dim=1
                        ).float()

                        target_masks_one_hot = F.one_hot(
                            target_masks,
                            num_classes=model.n_classes
                        )

                        target_masks_one_hot = target_masks_one_hot.permute(
                            0, 3, 1, 2
                        ).float()

                        dice_component_loss = dice_loss(
                            predicted_probabilities,
                            target_masks_one_hot,
                            multiclass=True
                        )

                        loss = (
                            pixel_classification_loss
                            + dice_component_loss
                        )
                optimizer.zero_grad(set_to_none=True)

                gradient_scaler.scale(loss).backward()

                gradient_scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clipping
                )

                gradient_scaler.step(optimizer)

                gradient_scaler.update()

                loss_value = loss.item()

                progress_bar.update(images.shape[0])

                global_step += 1

                epoch_loss += loss_value

                experiment.log(
                    {
                        '训练损失': loss_value,
                        '全局步数': global_step,
                        '训练轮数': epoch
                    }
                )

                progress_bar.set_postfix(
                    {
                        '当前批次损失': loss_value
                    }
                ) 
                validation_interval = (
                    number_of_training_samples // (5 * batch_size)
                )

                if validation_interval > 0:
                    is_validation_step = (
                        global_step % validation_interval == 0
                    )

                    if is_validation_step:
                        parameter_histograms = {}

                        for parameter_name, parameter_value in model.named_parameters():
                            parameter_contains_invalid_value = torch.any(
                                torch.isinf(parameter_value)
                                | torch.isnan(parameter_value)
                            )

                            if not parameter_contains_invalid_value:
                                parameter_histograms[
                                    '参数/' + parameter_name
                                ] = wandb.Histogram(
                                    parameter_value.detach().cpu()
                                )

                            if parameter_value.grad is not None:
                                gradient_contains_invalid_value = torch.any(
                                    torch.isinf(parameter_value.grad)
                                    | torch.isnan(parameter_value.grad)
                                )

                                if not gradient_contains_invalid_value:
                                    parameter_histograms[
                                        '梯度/' + parameter_name
                                    ] = wandb.Histogram(
                                        parameter_value.grad.detach().cpu()
                                    )

                        validation_dice_score = evaluate(
                            model,
                            validation_data_loader,
                            device,
                            amp
                        )

                        learning_rate_scheduler.step(
                            validation_dice_score
                        )

                        logging.info(
                            f'验证集 Dice 系数：{validation_dice_score}'
                        )

                        if model.n_classes == 1:
                            predicted_mask_for_visualization = (
                                torch.sigmoid(
                                    predicted_logits.squeeze(dim=1)
                                ) > 0.5
                            ).float()

                        else:
                            predicted_mask_for_visualization = (
                                predicted_logits.argmax(dim=1)
                            ).float()

                        experiment.log(
                            {
                                '学习率': optimizer.param_groups[0]['lr'],
                                '验证集 Dice': validation_dice_score,
                                '输入图像': wandb.Image(
                                    images[0].detach().cpu()
                                ),
                                '掩码': {
                                    '真实': wandb.Image(
                                        target_masks[0].float().detach().cpu()
                                    ),
                                    '预测': wandb.Image(
                                        predicted_mask_for_visualization[0]
                                        .detach()
                                        .cpu()
                                    )
                                },
                                '全局步数': global_step,
                                '训练轮数': epoch,
                                **parameter_histograms
                            }
                        )                     
        if save_checkpoint:
            checkpoints_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            checkpoint_state = model.state_dict()

            checkpoint_state['mask_values'] = dataset.mask_values

            checkpoint_file_name = (
                'checkpoint_epoch' + str(epoch) + '.pth'
            )

            checkpoint_path = (
                checkpoints_directory / checkpoint_file_name
            )

            torch.save(
                checkpoint_state,
                str(checkpoint_path)
            )

            logging.info(
                f'已保存第 {epoch} 轮的模型权重：{checkpoint_path}'
            )
def get_args():
    """解析训练程序的命令行参数"""

    parser = argparse.ArgumentParser(
        description='使用 U-Net 训练图像语义分割模型'
    )

    parser.add_argument(
        '--epochs',
        '-e',
        metavar='E',
        type=int,
        default=5,
        help='训练轮数'
    )

    parser.add_argument(
        '--batch-size',
        '-b',
        dest='batch_size',
        metavar='B',
        type=int,
        default=1,
        help='每个批次包含的图像数量'
    )

    parser.add_argument(
        '--learning-rate',
        '-l',
        dest='learning_rate',
        metavar='LR',
        type=float,
        default=1e-5,
        help='学习率'
    )

    parser.add_argument(
        '--load',
        '-f',
        dest='model_path',
        type=str,
        default=None,
        help='要加载的模型权重文件路径'
    )

    parser.add_argument(
        '--scale',
        '-s',
        dest='image_scale',
        type=float,
        default=0.5,
        help='图像缩放比例'
    )

    parser.add_argument(
        '--validation',
        '-v',
        dest='validation_percentage',
        type=float,
        default=10.0,
        help='验证集占全部数据的百分比'
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='启用自动混合精度训练'
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

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    args = get_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logging.info(f'使用计算设备：{device}')

    model = UNet(
        n_channels=3,
        n_classes=args.number_of_classes,
        bilinear=args.bilinear
    )

    model = model.to(
        memory_format=torch.channels_last
    )

    logging.info(
        f'网络结构配置：\n'
        f'    输入通道数：{model.n_channels}\n'
        f'    输出类别数：{model.n_classes}\n'
        f'    上采样方式：'
        f'{"双线性插值" if model.bilinear else "转置卷积"}'
    )

    if args.model_path:
        state_dict = torch.load(
            args.model_path,
            map_location=device
        )

        del state_dict['mask_values']

        model.load_state_dict(state_dict)

        logging.info(
            f'已加载模型权重：{args.model_path}'
        )

    model.to(device=device)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            image_scale=args.image_scale,
            validation_percentage=(
                args.validation_percentage / 100
            ),
            amp=args.amp
        )

    except torch.cuda.OutOfMemoryError:
        logging.error(
            '检测到显存不足，将启用梯度检查点来减少显存占用。'
            '这会降低训练速度。'
        )

        torch.cuda.empty_cache()

        model.use_checkpointing()

        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            image_scale=args.image_scale,
            validation_percentage=(
                args.validation_percentage / 100
            ),
            amp=args.amp
        )