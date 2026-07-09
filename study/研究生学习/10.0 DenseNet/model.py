from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int | None = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_rate = drop_rate

        if bn_size is None:
            self.bottleneck = None
            conv_in_channels = in_channels
        else:
            bottleneck_channels = bn_size * growth_rate
            self.bottleneck = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            )
            conv_in_channels = bottleneck_channels

        self.conv = nn.Sequential(
            nn.BatchNorm2d(conv_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bottleneck is not None:
            x = self.bottleneck(x)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)

        new_features = self.conv(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int | None = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Iterable[int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int | None = 4,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        num_classes: int = 1000,
        small_inputs: bool = False,
    ) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1].")

        if small_inputs:
            self.stem = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        channels = num_init_features
        blocks: list[nn.Module] = []
        block_config = tuple(block_config)
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            blocks.append(block)
            channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                out_channels = int(channels * compression)
                blocks.append(Transition(channels, out_channels))
                channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.norm_final = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)


@dataclass(frozen=True)
class CifarDenseNetConfig:
    depth: int
    growth_rate: int
    bottleneck: bool
    compression: float

    @property
    def layers_per_block(self) -> int:
        if self.bottleneck:
            if (self.depth - 4) % 6 != 0:
                raise ValueError("DenseNet-BC/B depth should satisfy (depth - 4) % 6 == 0.")
            return (self.depth - 4) // 6
        if (self.depth - 4) % 3 != 0:
            raise ValueError("Basic CIFAR DenseNet depth should satisfy (depth - 4) % 3 == 0.")
        return (self.depth - 4) // 3


def densenet_cifar(
    depth: int,
    growth_rate: int,
    num_classes: int = 10,
    bottleneck: bool = False,
    compression: float = 1.0,
    drop_rate: float = 0.0,
) -> DenseNet:
    config = CifarDenseNetConfig(
        depth=depth,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        compression=compression,
    )
    init_features = 2 * growth_rate if bottleneck else 16
    return DenseNet(
        growth_rate=growth_rate,
        block_config=(config.layers_per_block,) * 3,
        num_init_features=init_features,
        bn_size=4 if bottleneck else None,
        compression=compression,
        drop_rate=drop_rate,
        num_classes=num_classes,
        small_inputs=True,
    )


def densenet40_k12(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(40, 12, num_classes=num_classes, drop_rate=drop_rate)


def densenet100_k12(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(100, 12, num_classes=num_classes, drop_rate=drop_rate)


def densenet100_k24(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(100, 24, num_classes=num_classes, drop_rate=drop_rate)


def densenet_bc100_k12(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(100, 12, num_classes=num_classes, bottleneck=True, compression=0.5, drop_rate=drop_rate)


def densenet_bc250_k24(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(250, 24, num_classes=num_classes, bottleneck=True, compression=0.5, drop_rate=drop_rate)


def densenet_bc190_k40(num_classes: int = 10, drop_rate: float = 0.0) -> DenseNet:
    return densenet_cifar(190, 40, num_classes=num_classes, bottleneck=True, compression=0.5, drop_rate=drop_rate)


def densenet121(num_classes: int = 1000) -> DenseNet:
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)


def densenet169(num_classes: int = 1000) -> DenseNet:
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)


def densenet201(num_classes: int = 1000) -> DenseNet:
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes)


def densenet264(num_classes: int = 1000) -> DenseNet:
    return DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
