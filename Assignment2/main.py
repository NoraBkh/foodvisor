import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import get_model
from utils_metrics import *
from utils import get_dataset

PRINT_INTERVAL = 50
CUDA = False



def epoch(data, model, criterion, optimizer=None):
    """
    Make a pass (called epoch in English) on the data `data` with the
     model `model`. Evaluates `criterion` as loss.
     If `optimizer` is given, perform a training epoch using
     the given optimizer, otherwise, perform an evaluation epoch (no backward)
     of the model.
    """

    # indicates whether the model is in eval or train mode (some layers
    # behave differently in train and eval)
    model.eval() if optimizer is None else model.train()

    # objects to store metric averages
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    global loss_plot

    # we iterate over the batches of the dataset
    for i, (input, target) in enumerate(data):
        print("i===== ", i)
        if CUDA: # if we do GPU, switch to CUDA
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward if we are in "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics calculation
        prec1 = accuracy(output, target)[0]
       

        # update of averages
        avg_loss.update(loss.item())
        avg_acc.update(prec1.item())
        if optimizer:
            loss_plot.update(avg_loss.val)
        # info display
        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                  .format("EVAL" if optimizer is None else "TRAIN", i, len(data), loss=avg_loss,
                   top1=avg_acc))
            if optimizer:
                loss_plot.plot()

    # Viewing epoch info
    print('\n===============> Total'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f} %\t'
          .format(loss=avg_loss,
                  top1=avg_acc))

    return avg_acc, avg_loss


def main(params):

    # ex of params :
    #   {"batch_size": 128, "epochs": 5, "lr": 0.1, "path": '/tmp/datasets'}
    
    num_classes = 2
    # define model, loss, optim
    model = get_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.lr)

    if CUDA: # if we do GPU, switch to CUDA
        model = model.cuda()
        criterion = criterion.cuda()

    # We recover the data
    train, test = get_dataset(params.batch_size, params.path)

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()

    # We iterate on epochs
    for i in range(params.epochs):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Train phase
        top1_acc, loss = epoch(train, model, criterion, optimizer)
        # Evaluation Phase
        top1_acc_test, loss_test = epoch(test, model, criterion)
        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)
        # save model
        torch.save(model, 'Tomatomodel.pth')


if __name__ == '__main__':

    # Command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/nora/Téléchargements/test_foodvisor/Assignment2', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=100, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    main(args)

    input("done")
