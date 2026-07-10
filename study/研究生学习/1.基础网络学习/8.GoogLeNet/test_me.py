import os

import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import GoogLeNet, Inception
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')
ROOT_TEST_DATA = os.path.join(BASE_DIR, 'data_me', 'test')
CLASSES = ['cat', 'dog']


def test_data_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4879, 0.4545, 0.4167],
            std=[0.2596, 0.2524, 0.2552]
        )
    ])

    test_data = ImageFolder(ROOT_TEST_DATA, transform=transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=128,
                                      shuffle=False,
                                      num_workers=0)

    return test_dataloader, test_data.class_to_idx


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


if __name__ == "__main__":
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"没有找到模型文件：{BEST_MODEL_PATH}")

    model = GoogLeNet(Inception, num_classes=2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))

    test_dataloader, class_to_idx = test_data_process()
    print("类别映射：", class_to_idx)
    test_model_process(model, test_dataloader)

    image = Image.open(os.path.join(BASE_DIR, '狗1.jpg'))
    # image.show()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4879, 0.4545, 0.4167],
            std=[0.2596, 0.2524, 0.2552]
        )
    ])
    image = transform(image)
    #添加批次维度来适应模型输入
    image = image.unsqueeze(0).to('cuda')

    with torch.no_grad():
        model.eval()
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        print("预测结果：", CLASSES[pre_lab])



