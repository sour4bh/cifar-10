# import pdb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']


args_dataset = 'cifar10'
args_model = 'wideresnet'
args_batch_size = 128
args_epochs = 200
args_learning_rate = 0.1
args_data_augmentation = True
args_cutout = True
args_n_holes = 1
args_length = 16
args_no_cuda = False
args_cuda = True
args_seed = 42


args_cuda = not args_no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args_seed)
if args_cuda:
    torch.cuda.manual_seed(args_seed)

test_id = args_dataset + '_' + args_model

# print(args)

# Image Preprocessing

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args_data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args_cutout:
    train_transform.transforms.append(Cutout(n_holes=args_n_holes, length=args_length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])


num_classes = 10
train_dataset = datasets.CIFAR10(root='data/',
                                    train=True,
                                    transform=train_transform,
                                    download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args_batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args_batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args_model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args_model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args_learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)


scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

# filename = 'logs/' + test_id + '.csv'
# csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

try:
    checkpoint_fpath = 'cifar10_wideresnet100.pt'
    checkpoint = torch.load(checkpoint_fpath)
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn_optimizer.load_state_dict(checkpoint['optimizer'])
    begin = checkpoint['epoch']
except FileNotFoundError:
    print('starting over..')
    begin = 0


for epoch in range(args_epochs):
    if epoch <= begin:
        continue
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)
    checkpoint = {
    'epoch': epoch,
    'state_dict': cnn.state_dict(),
    'optimizer': cnn_optimizer.state_dict()
    }
    torch.save(checkpoint, 'epochs/' + test_id + f'{epoch}.pt')
    # row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    # csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')

