# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn import WideResNet
from models.wrn import WideResNet
from models.mstarconv import mstarVGG, MstarConvsss, CNN_12
import torchvision.models as models
import numpy as np

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'mstar'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv', 'wrn','newconv'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/mstar', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.Resize(32),trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('./data', download = True, train=True, transform=train_transform)
    test_data = dset.CIFAR10('./data', train=False, transform=test_transform)
    num_classes = 10
else:
    # train_data = dset.ImageFolder(root="/home/bwang30/study/ann/mstar_with_machine_learning/MSTAR-10/train",
    #                          transform=train_transform)
    # test_data = dset.ImageFolder('/home/bwang30/study/ann/mstar_with_machine_learning/MSTAR-10/test', transform=test_transform)
    num_classes = 2
    temp_transform = trn.Compose([trn.Resize([32,32]),trn.ToTensor()])    # 0~255 --> 0~1

    train_data = dset.ImageFolder('/home/bwang30/study/ann/mstar_with_machine_learning/MSTAR-10/newtask3train', temp_transform)

    test_data = dset.ImageFolder('/home/bwang30/study/ann/mstar_with_machine_learning/MSTAR-10/newtask3test', temp_transform)

    # train_data = dset.ImageFolder('/home/bwang30/study/ann/pytorch_template/download/mstar/oodtrain', temp_transform)

    # test_data = dset.ImageFolder('/home/bwang30/study/ann/pytorch_template/download/mstar/test', temp_transform)

    # train_loader = DataLoader(self.train_data, train_bs, True)
    # stest_loader = DataLoader(self.test_data, test_bs, True)
    
    ood_data = dset.ImageFolder(root="./data/textures/dtd/images",
                            transform=temp_transform)

    ood_data = lsun_loader.LSUN("./data/LSUN", classes='test',
                             transform=temp_transform)



calib_indicator = ''
if args.calibration:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    calib_indicator = '_calib'

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes)
elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

else:
    net = mstarVGG('VGG11')
    # net =  CNN_12(True)

start_epoch = 0

# Restore model if desired
if args.load != '':
    # for i in range(1000 - 1, -1, -1):
        # model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
        #                           '_baseline_epoch_' + str(i) + '.pt')
        # if os.path.isfile(model_name):
        #     net.load_state_dict(torch.load(model_name))
        #     print('Model restored! Epoch:', i)
        #     start_epoch = i + 1
        #     break
    model_name = args.load
    net.load_state_dict(torch.load(model_name))
    start_epoch = 200
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    for in_set, out_set in zip(train_loader, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data[:, :1, :, :])

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


feature_list = []
# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # forward
            output = net(data[:, :1, :, :])
            # print(output)
            # print(target)
            # if(target == 1):
            #     feature_list.append(output.cpu().numpy())

            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            # print(output)
            # print(target)
            # print(pred)
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)

    npfeature = np.array(feature_list)
    # print(npfeature.shape)
    # np.save("task2class5feature.npy",npfeature)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()


    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                            'oe_mstar32_newtask3_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                             'oe_mstar32_newtask3_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
