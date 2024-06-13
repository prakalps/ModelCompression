from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os, sys, shutil, time, random
from scipy.spatial import distance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
from data_loading import CarvanaDataset, BasicDataset
from tqdm import tqdm
from dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import logging
# import wandb
import torch.nn as nn
# from dice_score import dice_loss
from pathlib import Path

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CARVANA Images training')
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='UNet', type=str, help='architecture to use')
parser.add_argument('--depth', default=16, type=int, help='depth of the neural network')
# compress rate
parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_dist', type=float, default=0.1, help='the reducing ratio of pruning based on Distance')

# compress parameter
parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')

# pretrain model
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--use_precfg', dest='use_precfg', action='store_true', help='use precfg or not')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
dir_checkpoint="./"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.seed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)
    print_log("Pre cfg: {}".format(args.use_precfg), log)

    if args.dataset == 'CARVANA':
    # Create dataset
        try:
            dataset = CarvanaDataset(args.data_path+'/imgs', args.data_path+'/masks', scale=1)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(args.data_path+'/imgs', args.data_path+'/masks', scale=1)
        dataset=torch.utils.data.Subset(dataset, range(0,100))

        # 2. Split into train / validation partitions
        val_percent = 0.2
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
        # 3. Create data loaders
        loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
        test_loader = torch.utils.data.DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        

    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](n_channels=3, n_classes=2)
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    
    print_log("=> network :\n {}".format(model), log)

    if args.cuda:
        model.cuda()

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/anisha-gpu/code/Users/Anisha.Gupta/model-pruning-fpgm-unet-copy/logs/unet_pretrain/prune_precfg_epoch40_varience1'
            # args.pretrain_path = dir + '/checkpoint.pth.tar'
            args.pretrain_path = dir + '/MODEL.pth'
            print_log("Pretrain path: {}".format(args.pretrain_path), log)
        
        # pretrain = torch.load(args.pretrain_path)

        pretrain = torch.load(args.pretrain_path)
        # mask_values = pretrain.pop('mask_values', [0, 1])
        if args.use_state_dict:
            model.load_state_dict(pretrain)
        else:
            model = pretrain
        
        # if args.use_state_dict:
        #     model.load_state_dict(pretrain['state_dict'])
        # else:
        #     model = pretrain['state_dict']

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # model = models.vgg(dataset='cifar10', depth=16, cfg=checkpoint['cfg'])
            model = models.UNet(n_channels=1024, n_classes=1)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
            if args.cuda:
                model = model.cuda()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        time1 = time.time()
        test(test_loader, model, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    m = Mask(model)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))

    # val_acc_1 = test(test_loader, model, log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_acc_1 = evaluate(model, test_loader, device, amp=False)  # TMP
    print(" accu before is: %.3f %%" % val_acc_1.float())

    m.model = model

    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
    #    m.if_zero()
    m.do_mask()
    m.do_similar_mask()
    model = m.model
    #    m.if_zero()
    if args.cuda:
        model = model.cuda()
    val_acc_2 = evaluate(model, test_loader, device, amp=False)   # TMP
    print(" accu after is: %.3f %%" % val_acc_2.float())


    best_prec1 = 0.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    amp: bool = False,
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # global_step = 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # import torch
    torch.cuda.empty_cache()

    print("START", args.start_epoch)
    print("END", args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        train_model(model, train_loader, n_train, optimizer, epoch, log, grad_scaler, device, criterion, dataset) 
        prec1 = evaluate(model, test_loader, device, amp=False).float()
        # prec1 = test(test_loader, model, log)

        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = model
            m.if_zero()
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            m.do_mask()
            m.do_similar_mask()
            # small_filter_index.append(m.filter_small_index)
            # large_filter_index.append(m.filter_large_index)
            # save_obj(small_filter_index, 'small_filter_index_2')
            # save_obj(large_filter_index, 'large_filter_index_2')
            m.if_zero()
            model = m.model
            if args.cuda:
                model = model.cuda()
            # val_acc_2 = test(test_loader, model, log)
            val_acc_2 = evaluate(model, test_loader, device, amp=False).float()
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print(" Accuracy after epoch : %.3f %%" % best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        #     'cfg': model.cfg
        # }, is_best, filepath=args.save_path)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, filepath=args.save_path)

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     state_dict['mask_values'] = dataset.mask_values
        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
    
def train_model(
        model,
        train_loader, n_train,
        optimizer, epoch, log, grad_scaler,
        device, criterion, dataset,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 5. Begin training
    # for epoch in range(1, epochs + 1):
    print("Inside Train : ")
    print("Inside Train : Epochs: ",epochs)
    model.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']
            # print("IMAGE LENGTH :", len(images))

            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            # global_step += 1
            epoch_loss += loss.item()
            # experiment.log({
            #     'train loss': loss.item(),
            #     'step': global_step,
            #     'epoch': epoch
            # })
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            # division_step = (n_train // (5 * batch_size))
            # if division_step > 0:
            #     if global_step % division_step == 0:
            #         histograms = {}
            #         for tag, value in model.named_parameters():
            #             tag = tag.replace('/', '.')
            #             if not (torch.isinf(value) | torch.isnan(value)).any():
            #                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #             if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
            #                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        # logging.info('Validation Dice score: {}'.format(val_score))
                        # try:
                        #     experiment.log({
                        #         'learning rate': optimizer.param_groups[0]['lr'],
                        #         'validation Dice': val_score,
                        #         'images': wandb.Image(images[0].cpu()),
                        #         'masks': {
                        #             'true': wandb.Image(true_masks[0].float().cpu()),
                        #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         },
                        #         'step': global_step,
                        #         'epoch': epoch,
                        #         **histograms
                        #     })
                        # except:
                        #     pass

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     state_dict['mask_values'] = dataset.mask_values
        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')

def train(train_loader, model, optimizer, epoch, log, m=0):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_log('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), log)


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

def test(test_loader, model, log):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print_log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), log)
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 2]

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    def get_filter_similar_old(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            print('weight_vec.size', weight_vec.size())
            # distance using pytorch function
            similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            for x1, x2 in enumerate(filter_large_index):
                for y1, y2 in enumerate(filter_large_index):
                    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
                    pdist = torch.nn.PairwiseDistance(p=2)
                    # print('weight_vec[x2].size', weight_vec[x2].size())
                    similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
                    # print('weight_vec[x2].size after', weight_vec[x2].size())
            # more similar with other filter indicates large in the sum of row
            similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # for cos similar: get the filter index with largest similarity
            # similar_pruned_num = len(similar_sum) - similar_pruned_num
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_large_index]

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass
        return codebook

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = []
            if args.cuda:
                indices = torch.LongTensor(filter_large_index).cuda()
            else:
                indices = torch.LongTensor(filter_large_index)
                
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # print('similar_matrix 1',similar_matrix.cpu().numpy())
            # print('similar_matrix 2', similar_matrix_2)
            # # print('similar_matrix 3', similar_matrix_3)
            # result = np.absolute(similar_matrix.cpu().numpy() - similar_matrix_2)
            # print('result',result)
            # print('similar_matrix',similar_matrix.cpu().numpy())
            # print('similar_matrix_2', similar_matrix_2)
            # print('result', similar_matrix.cpu().numpy()-similar_matrix_2)
            # print('similar_sum',similar_sum)
            # print('similar_sum_2', similar_sum_2)
            # print('result sum', similar_sum-similar_sum_2)

            # for cos similar: get the filter index with largest similarity
            # similar_pruned_num = len(similar_sum) - similar_pruned_num
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_large_index]

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer, pre_cfg=True):
        if args.arch == 'UNet':
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 2]
            cfg_index = 0
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                self.distance_rate[index] = 1
                if len(item[1].size()) == 4:
                    print(item[1].size())
                    if not pre_cfg:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                        # print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = 1 - cfg[cfg_index] / item[1].size()[0]
                        

                        self.mask_index.append(index)
                        # print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg[cfg_index], item[1].size()[0],
                            #   self.distance_rate[index], )
                        # print("self.distance_rate", self.distance_rate)
                        cfg_index += 1
        # for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
        #     self.compress_rate[key] = rate_norm_per_layer
        #     self.distance_rate[key] = rate_dist_per_layer
        # different setting for  different architecture
        # if args.arch == 'resnet20':
        #     last_index = 57
        # elif args.arch == 'resnet32':
        #     last_index = 93
        # elif args.arch == 'resnet56':
        #     last_index = 165
        # elif args.arch == 'resnet110':
        #     last_index = 327
        # # to jump the last fc layer
        # self.mask_index = [x for x in range(0, last_index, 3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer, pre_cfg=args.use_precfg)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.cuda:
                    self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
