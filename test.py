#%%
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from cutout import Cutout
from wide_resnet import WideResNet
#%%
pt_file_path = 'cifar10_wideresnet93.pt'
args_batch_size = 128
num_classes = 10

def test(loader):
    model.eval()
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    return val_acc

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args_batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=1)

    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    model = model.cuda()
    
    checkpoint_fpath = pt_file_path
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    test_acc = test(test_loader)
    print('test_acc', test_acc)
# %%
