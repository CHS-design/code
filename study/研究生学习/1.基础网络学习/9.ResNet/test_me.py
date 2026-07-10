import os

import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn

from model import ResNet, Residual
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')
ROOT_TEST_DATA = os.path.join(BASE_DIR, 'data_me', 'test')
NUM_CLASSES = 2
BATCH_SIZE = 128


def build_model(num_classes=NUM_CLASSES):
    model = ResNet(Residual)
    model.fc[-1] = nn.Linear(model.fc[-1].in_features, num_classes)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4879, 0.4545, 0.4167],
            std=[0.2596, 0.2524, 0.2552]
        )
    ])


def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    output_classes = state_dict.get('fc.2.bias')
    if output_classes is not None and output_classes.shape[0] != NUM_CLASSES:
        raise ValueError(
            "当前 best_model.pth 是 {} 类输出，但 test_me.py 使用 cat/dog 二分类；请先运行 train_me.py 重新训练。"
            .format(output_classes.shape[0])
        )
    model.load_state_dict(state_dict)


def test_data_process():
    transform = get_transform()

    test_data = ImageFolder(ROOT_TEST_DATA, transform=transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=0)

    return test_dataloader, test_data


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_corrects = 0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y).item()
            test_num += test_data_x.size(0)

    test_acc = test_corrects / test_num
    print("测试的准确率为：{:.4f}".format(test_acc))
    return test_acc


def predict_image(model, image_path, class_to_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    image = Image.open(image_path).convert('RGB')
    image = get_transform()(image)
    image = image.unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pre_lab = torch.argmax(output, dim=1).item()

    print("单张图片：", image_path)
    print("预测结果：", idx_to_class[pre_lab])


if __name__ == "__main__":
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"没有找到模型文件：{BEST_MODEL_PATH}")

    model = build_model()
    load_model_weights(model, BEST_MODEL_PATH)

    test_dataloader, test_data = test_data_process()
    class_to_idx = test_data.class_to_idx
    print("类别映射：", class_to_idx)
    test_model_process(model, test_dataloader)

    sample_image_path = os.path.join(BASE_DIR, '狗1.jpg')
    if not os.path.exists(sample_image_path) and test_data.samples:
        sample_image_path = test_data.samples[2][0]
    if os.path.exists(sample_image_path):
        predict_image(model, sample_image_path, class_to_idx)
