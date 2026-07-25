import os

import torch
from model.unet_training import CE_Loss, Dice_loss, Focal_Loss
import numpy as np
from utils.utils import get_lr
from utils.utils_metrics import f_score
from torch.cuda.amp import autocast
import time

class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"

def pixel_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).float()
        correct_pixels = correct.sum().item()
        total_pixels = target.numel()
        return correct_pixels / total_pixels

def mean_accuracy(output, target, num_classes):
    """
    计算 Mean Pixel Accuracy (MPA).
    :param output: torch.Tensor, shape [N, C, H, W]
    :param target: torch.Tensor, shape [N, H, W]
    :param num_classes: int
    :return: float, mean pixel accuracy over valid classes
    """
    with torch.no_grad():
        # 取出每个像素的预测类别索引
        _, predicted = torch.max(output, dim=1)  # shape [N, H, W]

        accuracies = []
        for i in range(num_classes):
            # 找到该类别在标签和预测中的位置
            target_mask = (target == i)
            predicted_mask = (predicted == i)

            # 交集：预测正确的像素数（即 TP）
            intersection = torch.logical_and(target_mask, predicted_mask).sum().item()
            total = target_mask.sum().item()  # 标签中该类的总像素数

            if total > 0:
                acc = intersection / total
                accuracies.append(acc)
            # 如果该类别在 GT 中没有出现，则跳过，不计入平均

        # 防止所有类别都未出现
        if len(accuracies) == 0:
            return 0.0
        else:
            return sum(accuracies) / len(accuracies)


# 计算Mean IoU
def mean_iou(output, target, num_classes):
    """
    计算 mean IoU，只在 target 出现的类别中取平均
    """
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)  # (N, H, W)
        ious = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)

            intersection = torch.logical_and(target_mask, pred_mask).sum().item()
            union = torch.logical_or(target_mask, pred_mask).sum().item()

            if target_mask.sum().item() > 0:  # 只对 target 中存在的类求 IoU
                ious.append(intersection / union if union > 0 else 0.0)
        if len(ious) == 0:
            return 0.0
        return sum(ious) / len(ious)


# 计算Frequency Weighted IoU
def frequency_weighted_iou(output, target, num_classes):
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        ious = []
        frequencies = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)
            intersection = torch.logical_and(target_mask, pred_mask).sum().item()
            union = torch.logical_or(target_mask, pred_mask).sum().item()
            freq = target_mask.sum().item()
            frequencies.append(freq)
            ious.append((intersection / union) if union > 0 else 0.0)

        total = sum(frequencies)
        if total == 0:
            return 0.0
        fw_iou = sum(f * iou for f, iou in zip(frequencies, ious)) / total
        return fw_iou


def train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                             gpu_used, num_classes, scaler, epoch, train_epoch):
    # 设置类别权重参数。它是用来处理类别不平衡的问题的
    cls_weights = np.ones([num_classes], np.float32)
    epoch_loss = 0.0  # 总的训练损失

    total_f_score = 0

    # 切换为训练模式：
    # Dropout 使用随机失活；BatchNorm 使用当前批次更新统计量。
    model_train = model.train()

    # 模型与 imgs、pngs、labels 必须位于同一设备。
    # device 由 train_medical.py 决定，可能是 cuda 或 cpu。
    # .to(device) 不改变网络结构和张量形状，只改变参数存放位置。
    model_train = model_train.to(device)

    for iteration, batch in enumerate(train_loader):
        imgs, pngs, labels = batch  # 获取输入图像、标签和分割目标

        # 数据准备阶段：使用 `.to(device)` 自动将数据移到设备上
        weights = torch.tensor(cls_weights).to(device)  # 转换类别权重并移动到GPU
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清除之前的梯度

        # 混合精度训练
        if scaler is None:
            # 前向传播
            outputs = model_train(imgs)

            # 损失计算
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                # 如果使用Dice Loss，则加上Dice损失
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            # with torch.no_grad():
            #     # -------------------------------#
            #     #   计算f_score
            #     # -------------------------------#
            #     _f_score = f_score(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()  # 更新模型参数
        else:
            with autocast():
                # from torch.cuda.amp import autocast
                outputs = model_train(imgs)  # 通过模型获取预测结果

                #   损失计算
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                    # with torch.no_grad():
                    #     # -------------------------------#
                    #     #   计算f_score
                    #     # -------------------------------#
                    #     _f_score = f_score(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # 使用混合精度更新梯度
            scaler.update()  # 更新scaler

        # 累加训练损失和F-score
        epoch_loss += loss.item()
        # total_f_score += _f_score.item()

        # 打印标题（每个epoch开始时打印一次）
        if iteration == 0:  # 只在第一个 batch 打印标题
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}"
                  )

        # 每10个batch打印一次信息
        if iteration % 1 == 0:
            if len(train_loader) < 1:
                a = len(train_loader)
            else:
                a = 1

        Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
        batch_len = len("data_num") + 12 - len(str(f"{iteration + a}/{len(train_loader)}"))
        GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
        Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
        LR_len = len("LR") + 12 - len(str(f"{get_lr(optimizer):.8f}"))

        # 使用 \r 在同一行更新输出
        print(f"\r{epoch + 1}/{train_epoch}{' ' * Epoch_len}"
              f"{iteration + a}/{len(train_loader)}{' ' * batch_len}"
              f"{gpu_used:.2f} MB{' ' * GPU_len}"
              f"{loss.item():.8f}{' ' * Loss_len}"
              f"{get_lr(optimizer):.8f}{' ' * LR_len}"
              f"{imgs.shape[2]}", end='', flush=True)

    # 每个epoch结束后打印一次
    print(f"{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    # ➕ 返回平均loss
    return epoch_loss / len(train_loader)




def evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes):
    # 验证集为空时，后续平均值没有意义，应立即报错。
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty.")

    # CrossEntropyLoss 需要每个类别一个权重。
    # 目前所有类别权重相同，后续若发现类别严重不平衡再单独调整。
    cls_weights = torch.ones(num_classes, dtype=torch.float32, device=device)

    # eval(): Dropout 关闭随机失活，BatchNorm 使用已学习的统计量。
    # to(device): 确保模型与验证图像、标签位于同一设备。
    model_eval = model.eval().to(device)

    # 用样本数加权平均验证损失，避免最后一个小 batch 权重过大。
    total_loss = 0.0
    total_samples = 0

    # 混淆矩阵：行是真实类别，列是预测类别。
    # hist[2, 1] 表示“真实心肌被预测为左心室腔”的像素数量。
    hist = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device
    )

    # 验证阶段不构建梯度图，节省显存和计算。
    with torch.no_grad():
        for imgs, pngs, labels in val_loader:
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)

            # 输出形状：(B, 4, 256, 256)
            outputs = model_eval(imgs)

            # pngs 形状：(B, 256, 256)，每个像素的真实类别为 0~3。
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, cls_weights, num_classes)
            else:
                loss = CE_Loss(outputs, pngs, cls_weights, num_classes)

            # 若训练启用 Dice Loss，验证损失也必须使用相同组成。
            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # argmax 将每个像素的 4 个类别得分变成一个预测类别 ID。
            # predictions 形状：(B, 256, 256)，值为 0~3。
            predictions = torch.argmax(outputs, dim=1)

            # 仅统计合法真实标签，避免异常标签破坏混淆矩阵索引。
            valid = (pngs >= 0) & (pngs < num_classes)
            encoded = num_classes * pngs[valid] + predictions[valid]

            # 编码后的取值范围是 0~15；重塑后得到 4 x 4 混淆矩阵。
            hist += torch.bincount(
                encoded, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # 转为 float，便于后续计算比例指标。
    hist = hist.cpu().numpy().astype(np.float64)
    true_pixels = hist.sum(axis=1)
    pred_pixels = hist.sum(axis=0)
    intersection = np.diag(hist)
    union = true_pixels + pred_pixels - intersection

    # 每类 IoU = TP / (TP + FP + FN)。
    class_iou = np.divide(
        intersection,
        union,
        out=np.zeros(num_classes, dtype=np.float64),
        where=union > 0,
    )

    # Pixel Accuracy：所有预测正确像素 / 所有像素。
    pixel_acc = intersection.sum() / max(hist.sum(), 1.0)

    # Mean Accuracy：逐类召回率的平均，仅忽略验证集中完全不存在的类别。
    class_recall = np.divide(
        intersection,
        true_pixels,
        out=np.zeros(num_classes, dtype=np.float64),
        where=true_pixels > 0,
    )
    mean_acc = class_recall[true_pixels > 0].mean()

    # 标准 mIoU 包含背景；前景 mIoU 只平均 1、2、3 类，
    # 更适合观察三个心脏结构的实际分割效果。
    valid_iou = union > 0
    mean_iou = class_iou[valid_iou].mean()
    foreground_valid = valid_iou.copy()
    foreground_valid[0] = False
    foreground_miou = class_iou[foreground_valid].mean()

    # Frequency Weighted IoU：按各类别真实像素占比加权。
    class_frequency = true_pixels / max(hist.sum(), 1.0)
    fw_iou = (class_frequency * class_iou).sum()

    metrics = {
        "Pixel Accuracy": float(pixel_acc),
        "Mean Accuracy": float(mean_acc),
        "Mean IoU": float(mean_iou),
        "Foreground Mean IoU": float(foreground_miou),
        "Frequency Weighted IoU": float(fw_iou),
        "Loss": total_loss / total_samples,
        "Class IoU": class_iou.tolist(),
    }

    print(
        f"Val | loss={metrics['Loss']:.4f} | "
        f"mIoU={metrics['Mean IoU']:.4f} | "
        f"fg-mIoU={metrics['Foreground Mean IoU']:.4f}"
    )
    return metrics