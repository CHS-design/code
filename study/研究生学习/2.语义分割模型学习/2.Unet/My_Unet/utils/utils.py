import matplotlib.pyplot as plt


def plot_img_and_mask(image, mask):
    """显示输入图像和每个类别对应的预测掩码"""

    number_of_classes = int(mask.max()) + 1

    figure, axes = plt.subplots(
        1,
        number_of_classes + 1
    )

    axes[0].set_title('输入图像')
    axes[0].imshow(image)

    for class_index in range(number_of_classes):
        axes[class_index + 1].set_title(
            f'掩码（类别 {class_index}）'
        )
        axes[class_index + 1].imshow(
            mask == class_index
        )

    plt.xticks([])
    plt.yticks([])
    plt.show()
