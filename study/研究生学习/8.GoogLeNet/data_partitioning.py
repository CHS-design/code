import os
from shutil import copy # shutil 模块对文件进行复制、删除、移动等操作
import random # 随机数模块

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 数据集路径
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'data_cat_dog')
classes = [
    cla for cla in os.listdir(file_path)
    if os.path.isdir(os.path.join(file_path, cla))
] # 获取data_cat_dog文件夹下所有类别文件夹名
print(classes)

# 创建训练集、验证集、测试集文件夹
train_dir = os.path.join(base_dir,'data_me','train')
test_dir = os.path.join(base_dir, 'data_me', 'test')
mkfile(train_dir)
mkfile(test_dir)

#划分比例，训练集 : 测试集 = 9 : 1
split_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集、验证集和测试集
for cla in classes:
    cla_path = os.path.join(file_path,cla)  # 某一类别的子目录
    images = os.listdir(cla_path)  # images 列表存储了该 目录下所有图像的名称'
    num = len(images)
    test_index = random.sample(images, k=int(num * split_rate))  # 从images中随机采样
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in test_index:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(test_dir, cla)
            mkfile(new_path)  # 创建类别文件夹
            copy(image_path, new_path)  # 将选中的图像复制到新路径

        # 其余的图像保存在训练集train中
        else:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(train_dir, cla)
            mkfile(new_path)  # 创建类别文件夹
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")
