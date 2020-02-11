# imports 
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import datasets, transforms

from cutout import Cutout
from wide_resnet import WideResNet


# Control center
num_classes = 10
batch_size = 128
epochs = 200
seed = 0

LR_MILESTONES = [40, 60, 80, 90] # step down lr milestones
gamma = 0.2 #gamma for step lr 0.2 == 5x 
learning_rate = 0.1


data_augmentation = True
# cutout hyperparams
n_holes = 1
length = 16
# model - wideresnet hyperparams
depth = 28
widen_factor = 10
drop_rate = 0.3

def save_this(epoch, test_acc, accuracy, model, optimizer, scheduler, on_drive=True):
    checkpoint = {
    'epoch': epoch,
    'test_acc' : test_acc,
    'train_acc' : accuracy,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),    
    'scheduler' : scheduler.state_dict()}
    if on_drive:
        # save checkpoints to google drive 
        torch.save(checkpoint, '/content/drive/My Drive/Colab Notebooks/checkpoints/' + test_id + f'{epoch}.pt')
    else:
        # save checkpoints to local 
        torch.save(checkpoint, 'epochs/' +  test_id +  f'{epoch}.pt')

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
    model.train()
    return val_acc

if __name__ == "__main__":
    cuda = True
    cudnn.benchmark = True 
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    test_id = 'cifar10' + '_' + model

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    if data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # cutout augmentation     
    train_transform.transforms.append(Cutout(n_holes=n_holes, length=length)) # cutout augemntation 

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.CIFAR10(root='data/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)

    model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=drop_rate)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0005)

    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=gamma)

 

    try:
        checkpoint_fpath = 'cifar-10/cifar10_wideresnet79.pt'
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.2, last_epoch=checkpoint['epoch'])
        begin = checkpoint['epoch']
        # print('test_acc :', checkpoint['test_acc'], 'train_acc :', checkpoint['train_acc'])
        # print('last_lr :', checkpoint['scheduler']['_last_lr'])
    except FileNotFoundError:
        # print('starting over..')
        begin = -1



    best_acc = 0
    for epoch in range(epochs):
        if epoch <= begin:
            # scheduler.step()
            continue
        
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()
            pred = model(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
            _lr=optimizer.param_groups[0]["lr"]

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' %(accuracy),
                lr='%.2E'%(_lr))

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            save_this(epoch, test_acc, accuracy, model, optimizer, scheduler, on_drive=False)
        
        if epoch % 10 == 0:
            save_this(epoch, test_acc, accuracy, model, optimizer, scheduler, on_drive=True)
        
        _lr = optimizer.param_groups[0]["lr"]
        with open('log.csv', 'a') as f:
            f.write(f"epoch: {str(epoch)}, train_acc: {str(accuracy)}, test_acc: {str(test_acc)}, lr:{'%.2E'%(_lr)}" + '\n')
        
        scheduler.step()
        # end loop