import torch
from torch import Tensor


def dice_coeff(
    predicted_mask: Tensor,
    target_mask: Tensor,
    reduce_batch_first=False,
    epsilon=1e-6
):
    """计算二分类掩码的 Dice 系数"""

    if predicted_mask.size() != target_mask.size():
        raise AssertionError(
            '预测掩码与真实掩码的形状必须一致'
        )

    if predicted_mask.dim() == 2:
        summation_dimensions = (-1, -2)

    elif reduce_batch_first:
        summation_dimensions = (-1, -2, -3)

    else:
        summation_dimensions = (-1, -2)

    intersection = 2 * (
        predicted_mask * target_mask
    ).sum(dim=summation_dimensions)

    predicted_and_target_sum = (
        predicted_mask.sum(dim=summation_dimensions)
        + target_mask.sum(dim=summation_dimensions)
    )

    predicted_and_target_sum = torch.where(
        predicted_and_target_sum == 0,
        intersection,
        predicted_and_target_sum
    )

    dice = (
        intersection + epsilon
    ) / (
        predicted_and_target_sum + epsilon
    )

    return dice.mean()

def multiclass_dice_coeff(
    predicted_masks: Tensor,
    target_masks: Tensor,
    reduce_batch_first=False,
    epsilon=1e-6
):
    """计算多类别分割任务的平均 Dice 系数"""

    flattened_predicted_masks = predicted_masks.flatten(
        start_dim=0,
        end_dim=1
    )

    flattened_target_masks = target_masks.flatten(
        start_dim=0,
        end_dim=1
    )

    return dice_coeff(
        flattened_predicted_masks,
        flattened_target_masks,
        reduce_batch_first,
        epsilon
    )

def dice_loss(
    predicted_masks: Tensor,
    target_masks: Tensor,
    multiclass=False
):
    """计算用于训练的 Dice 损失"""

    if multiclass:
        dice_function = multiclass_dice_coeff
    else:
        dice_function = dice_coeff

    dice_score = dice_function(
        predicted_masks,
        target_masks,
        reduce_batch_first=True
    )

    return 1 - dice_score