import argparse
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter

from data import HSTrainingData
from data import HSTestData

from loss import HybridLoss
from loss import MixLoss

from metrics import quality_assessment

from common import *
import numpy as np

from Ours import GELIN
from thop import profile
from loss import HLoss
def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for LPSR-Net network")
    main_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    main_parser.add_argument("--epochs", type=int, default=300, help="epochs, default set to 20")   # change
    main_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 24")
    main_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    main_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    main_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    main_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")   # change


    # datasets
    main_parser.add_argument("--dataset_name", type=str, default="Cave",
                              help="dataset_name, default set to dataset_name")        # change
    main_parser.add_argument("--n_scale", type=int, default=4, help="the img size to super resolution") # change

    # network
    main_parser.add_argument("--model_title", type=str, default="RSSINet",
                              help="model_title, default set to model_title")
    main_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    main_parser.add_argument("--n_blocks", type=int, default=6, help="n_blocks, default set to 6")   # change
    main_parser.add_argument("--n_layers", type=int, default=2, help="n_blocks, default set to 6")
    main_parser.add_argument("--n_subs", type=int, default=4, help="n_subs, default set to 8")      # change
    main_parser.add_argument("--n_ovls", type=int, default=1, help="n_ovls, default set to 1")
    main_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    train(args)


# global settings
#resume = True

resume = 0
log_interval = 10

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    if args.dataset_name == 'Cave':
        train_path = '/data2/CaveDataMat/trains/'
        test_path  = '/data2/CaveDataMat/tests/'
    elif args.dataset_name == 'Chikusei':
        train_path = '/data2/ChikuseiData/trains/'
        test_path  = '/data2/ChikuseiData/tests/'

    train_set = HSTrainingData(image_dir=train_path, n_scale=args.n_scale, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_set = HSTestData(image_dir=test_path, n_scale=args.n_scale)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.dataset_name == 'Cave' or args.dataset_name == 'Harvard' or args.dataset_name == 'ICVL':
        colors = 31
    elif args.dataset_name == 'Pavia':
        colors = 102
    elif args.dataset_name == 'Chikusei':
        colors = 128
    else:
        colors = 33
    print('===> Building model')
     

    # For CAVE
    net = GELIN(n_feats=args.n_feats,n_colors=colors,kernel_size=3,pooling_r=2,n_subs=8,n_ovls=2,blocks=8,scale=args.n_scale)

    # For Chikusei
    
    #net = GELIN(n_feats=args.n_feats,n_colors=colors,kernel_size=3,pooling_r=2,n_subs=32,n_ovls=8,blocks=6,scale=args.n_scale)

    print('# parameters:', sum(param.numel() for param in net.parameters())) 
    model_title = args.dataset_name + "_" + args.model_title + '_n_scale' + str(args.n_scale)
    model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(3800) + ".pth"
    args.model_title = model_title
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()
    
    # loss functions to choose
    L1_loss = torch.nn.L1Loss()
    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter('./runs/' + model_title + '_' + str(time.ctime()))

    print('===> Start training')
    epoch_meter = meter.AverageValueMeter()
    my_loss = HLoss(0.3,0.1)
    
    for e in range(start_epoch, args.epochs):
        epoch_meter.reset()
        for iteration, (x, lms, gt) in enumerate(train_loader):

            adjust_learning_rate(args.learning_rate, optimizer, e+1)
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y = net(x, lms)

            loss_all = my_loss(y, gt)
            epoch_meter.add(loss_all.item())
            loss_all.backward()
            optimizer.step()
        writer.add_scalar('scalar/train_loss', epoch_meter.value()[0], e + 1)

        # save model weights at checkpoints every 10 epochs
        if (e + 1) % log_interval ==0:
            print("epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
            print("Training Loss: {:.6f}".format(epoch_meter.value()[0]))

            writer.add_image('image/epoch' + str(e) + 'ms', np.expand_dims(x.cpu().numpy()[0, (9), :, :],axis=0))
            writer.add_image('image/epoch' + str(e) + 'pre', np.expand_dims(y.cpu().detach().numpy()[0, (9), :, :],axis=0))
            writer.add_image('image/epoch' + str(e) + 'lms', np.expand_dims(lms.cpu().numpy()[0, (9), :, :],axis=0))
            writer.add_image('image/epoch' + str(e) + 'gt', np.expand_dims(gt.cpu().numpy()[0, (9), :, :],axis=0))

            test_loss = validate(args, test_loader, net, L1_loss)
            writer.add_scalar('scalar/avg_test_loss', test_loss, e + 1)
            save_checkpoint(args, net, e + 1)

          
    ## Save the testing results
    print("Running testset")
    print('===> Start testing')
    net.eval().cuda()
    with torch.no_grad():
        output = []
        test_number = 0
        for i, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = net(ms, lms)

            ###################
            # TangZhenjie
            ###################
            # CAVE is 9 for visualization
            writer.add_image('image/test' + str(i) + 'gt', gt.squeeze().cpu().numpy()[(80), :, :])
            writer.add_image('image/test' + str(i) + 'ms', ms.squeeze().cpu().numpy()[(80), :, :])
            writer.add_image('image/test' + str(i) + 'pre', y.squeeze().cpu().numpy()[(80), :, :])
            writer.add_image('image/test' + str(i) + 'lms', lms.squeeze().cpu().numpy()[(80), :, :])

            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # print the test result
    print(indices)
def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp
# poly learning rate
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))
    
def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3600))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            # y = model(ms)
            y = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]
if __name__ == "__main__":
    main()
