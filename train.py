# -*- coding:utf-8  -*-
import math
import time
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import torch
import random
from model.Ours import OursModel
from model.PNN import PNNModel
from model.PanNet import PanNetModel
from model.TFNet import TFNetModel
from model.MSDCNN import MSDCNNModel
from model.SRPPNN import SRPPNNModel
from model.PDIN import PDINModel
from model.DCNN import DCNNModel
from model.PDINSAM import PDINSAMModel
from model.PDINFS import PDINFSModel
from model.PDIMN import PDIMNModel
from model.SSPN import SSPNModel
from model.SSPNT import SSPNTModel
from model.PPN import PPNModel

from model.HFN import HFNModel
from model.HFNCA import HFNCAModel
from model.HFNBaseline import HFNBaselineModel
from options import config_hfnca as hfnca_cfg
from model.PReN import PReModel
from options import config_pren as pren_cfg
from options import config_pnn as pnn_cfg
from options import config_msdcnn as msdcnn_cfg
from options import config_srppnn as srppnn_cfg
from options import config_pannet as pannet_cfg
from options import config_tfnet as tfnet_cfg
from options import config_pdin as pdin_cfg
from options import config_dcnn as dcnn_cfg
from options import config_pdinsam as pdinsam_cfg
from options import config_hfn as hfn_cfg
from options import config_pdinfs as pdinfs_cfg
from options import config_pdimn as pdimn_cfg
from options import config_sspn as sspn_cfg
from options import config_sspnt as sspnt_cfg
from options import config_ppn as ppn_cfg

from options import config_hfnbaseline as hfnbaseline_cfg

from data import PsDataset, Get_DataSet
from utils import *
import config as cfg
import metrics
import warnings

from matplotlib import pyplot as plt
import math
# import gdal
from osgeo import gdal
from tqdm import tqdm
from thop import profile
from ptflops import get_model_complexity_info
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings

warnings.filterwarnings("ignore")


def get_dataset(args):
    data_train = PsDataset(args, apath=args.dataDir, isAug=args.isAug,
                           isUnlabel=args.isUnlabel)  # PsRamDataset(apath=cfg.dataDir, isUnlabel=cfg.isUnlabel)#LmdbDataset()#
    if args.isEval:
        train_data, test_data = Get_DataSet(data_train, [0.8, 0.2])
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batchSize,
                                                 drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                                 pin_memory=True)
        dataloader2 = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize,
                                                  drop_last=True, shuffle=False, num_workers=int(args.nThreads),
                                                  pin_memory=True)
        return dataloader, dataloader2
    else:
        dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                                 drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                                 pin_memory=True)
        return dataloader


def evalOrSaveBest(net, dataloader, best_eval_index, args):
    step = 0
    current_eval_index = 0

    #
    val_loss = 0
    val_rmse = 0
    val_psnr = 0
    net.eval()
    with torch.no_grad():

        for batch, (im_lr, im_hr, im_fr) in enumerate(dataloader):
            img_low_resolution = Variable(im_lr.cuda(), volatile=False)
            img_high_resolution = Variable(im_hr.cuda())
            img_pansherpen = Variable(im_fr.cuda())
            input_dict = {'A_1': img_low_resolution,
                          'A_2': img_high_resolution,
                          'B': img_pansherpen}
            net.set_input(input_dict)
            net.forward()

            fake_B = net.fake_B.cpu().detach().numpy() * args.data_range
            real_B = net.real_B.cpu().detach().numpy() * args.data_range

            current_batch_eval_index = metrics.get_rmse(real_B, fake_B)

            current_eval_index += current_batch_eval_index

            print('Valing: {}'.format(step), 'current_rmse: {}'.format(current_batch_eval_index / args.batchSize))
            step += 1

            val_loss += net.loss_G
            val_rmse += current_batch_eval_index
            val_psnr += metrics.psnr(fake_B, real_B, dynamic_range=args.data_range)

    # print(len(dataloader))
    current_eval_index = current_eval_index / len(dataloader) / args.batchSize

    print('val_rmse=', current_eval_index, 'best_rmse', best_eval_index)
    if current_eval_index < best_eval_index:
        print('better than best_rmse=', best_eval_index, 'save to best')
        best_eval_index = current_eval_index
        net.save_networks('best')
    return best_eval_index, val_loss / len(dataloader), val_rmse / len(dataloader), val_psnr / len(dataloader)


def gdal_write(output_file, array_data):
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    c, h, w = array_data.shape
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file, w, h, c, datatype)

    for i in range(c):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array_data[i, :, :])


def train(args):
    log = LossLog(args)

    #  select network
    if args.model == 'ours':
        cycle_gan = OursModel()
    elif args.model == 'PNN':
        cycle_gan = PNNModel()
    elif args.model == 'PanNet':
        cycle_gan = PanNetModel()
    elif args.model == 'TFNet':
        cycle_gan = TFNetModel()
    elif args.model == 'PSGAN':
        cycle_gan = PSGANModel()
    elif args.model == 'MSDCNN':
        cycle_gan = MSDCNNModel()
    elif args.model == 'SRPPNN':
        cycle_gan = SRPPNNModel()
    elif args.model == 'PDIN':
        cycle_gan = PDINModel()
    elif args.model == 'DCNN':
        cycle_gan = DCNNModel()
    elif args.model == 'PDINSAM':
        cycle_gan = PDINSAMModel()
    elif args.model == 'PDINFS':  #
        cycle_gan = PDINFSModel()
    elif args.model == 'PDIMN':  #
        cycle_gan = PDIMNModel()
    elif args.model == 'HFN':  #
        cycle_gan = HFNModel()
        print(args.B)
        print(args.features)
    elif args.model == 'HFNCA':
        cycle_gan = HFNCAModel()
        print(args.B)
        print(args.features)

    elif args.model == 'HFNSA':
        cycle_gan = HFNSAModel()
        print(args.B)
        print(args.features)


    elif args.model == 'HFNBaseline':
        cycle_gan = HFNBaselineModel()
        print(args.B)
        print(args.features)

    elif args.model == 'HFNLSTM':
        cycle_gan = HFNLSTMModel()
        print(args.B)
        print(args.features)

    elif args.model == 'PREN':
        cycle_gan = PReModel()

    elif args.model == 'SSPN':
        cycle_gan = SSPNModel()

    elif args.model == 'SSPNT':
        cycle_gan = SSPNTModel()

    elif args.model == 'PPN':
        cycle_gan = PPNModel()

    cycle_gan.initialize(args)
    print(cycle_gan)
    # cycle_gan.cuda()

    cycle_gan.setup()
    if args.scale == 6:
        input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
        input2 = torch.randn(1, args.pan_channel, 384, 384).cuda()
    else:
        input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
        input2 = torch.randn(1, args.pan_channel, 256, 256).cuda()
    flop, para = profile(cycle_gan.netG, inputs=(input1, input2,))
    print("%.2fM" % (flop / 1e6), "%.2fM" % (para / 1e6))

    print('Total params: %.2fM' % (sum(p.numel() for p in cycle_gan.netG.parameters()) / 1000000.0))

    # load data
    if args.isEval:
        dataloader, dataloader2 = get_dataset(args)
    else:
        dataloader = get_dataset(args)

    batch_iter = 0  # iterations
    lr_decay_iters_idx = 0

    cycle_gan.train()

    best_psnr = 999999

    # mse
    mse = nn.MSELoss()

    epoch_iter_nums = len(dataloader)

    total_iter_nums = epoch_iter_nums * args.epochs

    avg_loss = 0

    avg_rmse = 0

    avg_val_loss = 0

    avg_val_rmse = 0

    loss_history = []

    rmse_history = []
    #
    val_loss_history = []
    #
    val_rmse_history = []

    avg_psnr = 0

    avg_val_psnr = 0

    psnr_history = []
    #
    val_psnr_history = []

    epoch_history = [i + 1 for i in range(args.epochs)]

    for epoch in range(args.which_epoch + 1, args.epochs):
        iter_data_time = time.time()

        for batch, (im_lr, im_hr, im_fr) in enumerate(dataloader):
            iter_start_time = time.time()

            img_low_resolution = Variable(im_lr.cuda(), volatile=False)
            img_high_resolution = Variable(im_hr.cuda())
            img_pansherpen = Variable(im_fr.cuda())
            input_dict = {'A_1': img_low_resolution,
                          'A_2': img_high_resolution,
                          'B': img_pansherpen}

            cycle_gan.set_input(input_dict)
            cycle_gan.optimize_parameters()

            losses = cycle_gan.get_current_losses()

            for k, v in losses.items():
                if k == 'G':
                    avg_loss += v

            #
            avg_rmse += metrics.get_rmse(cycle_gan.fake_B.detach().cpu().numpy(),
                                         cycle_gan.real_B.detach().cpu().numpy())
            avg_psnr += metrics.psnr(cycle_gan.fake_B.detach().cpu().numpy() * args.data_range,
                                     cycle_gan.real_B.detach().cpu().numpy() * args.data_range,
                                     dynamic_range=args.data_range)

            if (batch_iter + 1) % args.print_freq == 0:
                t = (time.time() - iter_start_time) / args.batchSize
                t_data = iter_start_time - iter_data_time
                log.print_current_losses(epoch, batch, epoch_iter_nums, batch_iter, total_iter_nums, losses, t, t_data)

            batch_iter += 1

        #
        loss_history.append(avg_loss / epoch_iter_nums)
        avg_loss = 0
        rmse_history.append(avg_rmse / epoch_iter_nums)
        avg_rmse = 0
        psnr_history.append(avg_psnr / epoch_iter_nums)
        avg_psnr = 0

        change_infos = cycle_gan.update_learning_rate(decay_factor=args.lr_decay_factor)

        for info in change_infos:
            log.print_change_learning_rate(epoch, info['name'], info['old_lr'], info['new_lr'])

        if args.isEval:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = evalOrSaveBest(net=cycle_gan, dataloader=dataloader2,
                                                                                 best_eval_index=best_psnr, args=args)
        else:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = 0, 0, 0, 0
        val_loss_history.append(avg_val_loss)
        val_rmse_history.append(avg_val_rmse)
        val_psnr_history.append(avg_val_psnr)

        if (epoch + 1) % args.save_epoch_freq == 0:
            cycle_gan.save_networks(epoch)

    print('final best =', best_psnr)
    plt.plot(epoch_history, loss_history, 'r', label='Training loss')
    plt.plot(epoch_history, val_loss_history, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'loss.png')
    plt.savefig(path, dpi=300)

    plt.figure()
    plt.plot(epoch_history, rmse_history, 'r', label='Training rmse')
    plt.plot(epoch_history, val_rmse_history, 'b', label='Validation rmse')
    plt.title('Training and validation rmse')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'accuracy.png')
    plt.savefig(path, dpi=300)

    plt.figure()
    plt.plot(epoch_history, psnr_history, 'r', label='Training psnr')
    plt.plot(epoch_history, val_psnr_history, 'b', label='Validation psnr')
    plt.title('Training and validation psnr')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'psnr.png')
    plt.savefig(path, dpi=300)

    # plt.show()

    #
    np.save(os.path.join(cycle_gan.save_dir, 'epochs.npy'), np.array(epoch_history))
    np.save(os.path.join(cycle_gan.save_dir, 'losses.npy'), np.array(loss_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_losses.npy'), np.array(val_loss_history))
    np.save(os.path.join(cycle_gan.save_dir, 'rmses.npy'), np.array(rmse_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_rmses.npy'), np.array(val_rmse_history))
    np.save(os.path.join(cycle_gan.save_dir, 'psnrs.npy'), np.array(psnr_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_psnrs.npy'), np.array(val_psnr_history))


def set_seed(seed):
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    warnings.filterwarnings("ignore")
    parser.add_argument("-cfg", "--trainCfg", type=str, default='sspnt', help="")
    opt = parser.parse_args()

    args = None
    if opt.trainCfg == 'pnn':
        args = pnn_cfg
    elif opt.trainCfg == 'dcnn':
        args = dcnn_cfg
    elif opt.trainCfg == 'msdcnn':
        args = msdcnn_cfg
    elif opt.trainCfg == 'tfnet':
        args = tfnet_cfg
    elif opt.trainCfg == 'srppnn':
        args = srppnn_cfg
    elif opt.trainCfg == 'pannet':
        args = pannet_cfg
    elif opt.trainCfg == 'pdinsam':
        args = pdinsam_cfg
    elif opt.trainCfg == 'hfn':
        args = hfn_cfg

    elif opt.trainCfg == 'hfnca':
        args = hfnca_cfg
    elif opt.trainCfg == 'hfnsa':
        args = hfnsa_cfg
    elif opt.trainCfg == 'hfnbaseline':
        args = hfnbaseline_cfg
    elif opt.trainCfg == 'hfnlstm':
        args = hfnlstm_cfg


    elif opt.trainCfg == 'pren':
        args = pren_cfg


    elif opt.trainCfg == 'sspn':
        args = sspn_cfg

    elif opt.trainCfg == 'sspnt':
        args = sspnt_cfg

    elif opt.trainCfg == 'ppn':
        args = ppn_cfg

    print('load cfg from ', opt.trainCfg)
    print(args.seed)
    set_seed(args.seed)
    train(args)
