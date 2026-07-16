import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_coeff, multiclass_dice_coeff


@torch.inference_mode()
def evaluate(network, data_loader, device, amp):
    """在验证集上计算平均 Dice 系数"""

    network.eval()

    number_of_validation_batches = len(data_loader)
    total_dice_score = 0

    if device.type == 'mps':
        autocast_device_type = 'cpu'
    else:
        autocast_device_type = device.type

    with torch.autocast(
        device_type=autocast_device_type,
        enabled=amp
    ):
        for batch in tqdm(
            data_loader,
            total=number_of_validation_batches,
            desc='验证中',
            unit='批次',
            leave=False
        ):
            images = batch['image']
            target_masks = batch['mask']

            images = images.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last
            )

            target_masks = target_masks.to(
                device=device,
                dtype=torch.long
            )

            predicted_logits = network(images)

            if network.n_classes == 1:
                minimum_mask_value = target_masks.min().item()
                maximum_mask_value = target_masks.max().item()

                if minimum_mask_value < 0:
                    raise AssertionError(
                        '二分类真实掩码的类别索引不能小于 0'
                    )

                if maximum_mask_value > 1:
                    raise AssertionError(
                        '二分类真实掩码的类别索引不能大于 1'
                    )

                predicted_masks = torch.sigmoid(
                    predicted_logits.squeeze(dim=1)
                )

                predicted_masks = (
                    predicted_masks > 0.5
                ).float()

                total_dice_score += dice_coeff(
                    predicted_masks,
                    target_masks,
                    reduce_batch_first=False
                )

            else:
                minimum_mask_value = target_masks.min().item()
                maximum_mask_value = target_masks.max().item()

                if minimum_mask_value < 0:
                    raise AssertionError(
                        '真实掩码的类别索引不能小于 0'
                    )

                if maximum_mask_value >= network.n_classes:
                    raise AssertionError(
                        '真实掩码中存在超出类别范围的索引'
                    )

                target_masks = F.one_hot(
                    target_masks,
                    num_classes=network.n_classes
                )

                target_masks = target_masks.permute(
                    0, 3, 1, 2
                ).float()

                predicted_class_indices = predicted_logits.argmax(
                    dim=1
                )

                predicted_masks = F.one_hot(
                    predicted_class_indices,
                    num_classes=network.n_classes
                )

                predicted_masks = predicted_masks.permute(
                    0, 3, 1, 2
                ).float()

                foreground_predicted_masks = predicted_masks[:, 1:, :, :]
                foreground_target_masks = target_masks[:, 1:, :, :]

                total_dice_score += multiclass_dice_coeff(
                    foreground_predicted_masks,
                    foreground_target_masks,
                    reduce_batch_first=False
                )

    network.train()

    if number_of_validation_batches == 0:
        return 0

    return total_dice_score / number_of_validation_batches