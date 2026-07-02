from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt



train_dataset = FashionMNIST(root='./data',
                             train=True,
                             transform=transforms.Compose(transforms=[transforms.Resize(224),
                                                                      transforms.ToTensor()]),
                             download=True)


train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)




# 使用for循环遍历训练数据加载器train_loader
# enumerate函数会返回当前步骤step和对应的批数据(b_x, b_y)
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x=b_x.squeeze().numpy()
batch_y=b_y.numpy()
class_label = train_dataset.classes
# print(class_label)

plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
