#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math
from random import randrange
from tqdm import tqdm

import shutil

import setproctitle

import densenet
import make_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--augment', type=str, default='')
    parser.add_argument('--display')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

########################################################################################
#
#   DATA MUNGING
#
########################################################################################

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    # Data augmentation/regularization methods are specified in the command line, e.g.
    #
    # $ python train.py --augment=cutout
    # $ python train.py --augment=cutout,negative,quadrant
    #
    # Online transformations are taken care of in the training loop.
    tf_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if 'cutout' in args.augment:
        tf_list.append(transforms.Lambda(
                lambda img: cutout(img)
            ))
    if 'negative' in args.augment:
        tf_list.append(transforms.Lambda(
                lambda img: negative(img)
            ))
    tf_list.append(normTransform)
    trainTransform = transforms.Compose(tf_list) # Note that normTransform still needs to be added after.

    # Load data into training and testing batches.
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.display:
        trainLoader = DataLoader(
            dset.CIFAR100(root='cifar', train=True, download=True, transform=trainTransform),
            batch_size=args.batchSz, shuffle=True, **kwargs)
        for (batch, target) in trainLoader:
            batch[0].save('sample_image.jpeg')  # save sample image
    
    # Prepare to feed data into the network, applying normalizing transform now.
    trainTransform = transforms.Compose([
            trainTransform,
            normTransform
        ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    trainLoader = DataLoader(
        dset.CIFAR100(root='cifar', train=True, download=True,
                    transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR100(root='cifar', train=False, download=True,
                    transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

########################################################################################
#
#   DENSENET
#
########################################################################################

    # Create the DenseNet.
    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=100)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    # Preprocess and augment data
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))
    
    trainF.close()
    testF.close()

# converts a 3x32x32 Tensor to an RGB image file
def untransform(data, mean, std):
    filename = 'sample_image.jpeg' 
    data = [d*s + m for d, m, s in zip(data, mean, std)]
    print(data)
    data = torch.FloatTensor(data)
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    tf(data).save(filename)

# converts a 32x32 input image to an 8x8 quadrant
def quadrant(data):
    for mat in data:
        dim = 8
        quadrant = dim * np.random.randint(4)
        mat = mat[:, [quadrant, quadrant+dim-1]]
    return data

def cutout(data):
    size = 8
    for mat in data:
        # top-left corner of cutout
        # can be past len(mat) - size, giving smaller than size * size cutout
        cx, cy = randrange(0, len(mat)), randrange(0, len(mat[0]))
        # if size is even, center leans right and down
        for i in range(int(-size / 2), round(size / 2)):
            if cx + i < 0:
                continue
            if cx + i == len(mat):
                break
            for j in range(int(-size / 2), round(size / 2)):
                if cy + j < 0:
                    continue
                if cy + j == len(mat):
                    break
                mat[cx + i][cy + j] = 0
    return data

# color ranges from ~-2 to 2, so flipping sign
def negative(data):
    for mat in data:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = -mat[i][j]
    return data

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
