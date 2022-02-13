# -*- coding:utf-8  -*-
import numpy as np
import cv2
import os
import glob
from scipy import signal

# DL-methods
from model.PNN import PNNModel
from model.PanNet import PanNetModel
from model.Ours import OursModel
from model.TFNet import TFNetModel
from model.MSDCNN import MSDCNNModel
from model.DCNN import DCNNModel
from model.PDINSAM import PDINSAMModel
from model.PDINFS import PDINFSModel
from model.PDIMN import PDIMNModel
from model.PDIN import PDINModel
from model.SRPPNN import SRPPNNModel
from model.SSPN import SSPNModel
from model.HFNBaseline import HFNBaselineModel
from model.PReN import PReModel
from options import config_pren as pren_cfg
import cupy as cp
from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from options import config_hfn
from options import config_hfnbaseline as hfnbaseline_cfg
from options import config_sspn as sspn_cfg


# import gdal
from osgeo import gdal

from tqdm import tqdm
import torch
import time
from thop import profile
import argparse

def gdal_read(input_file):
   
    dataset = gdal.Open(input_file)
    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    data = dataset.ReadAsArray() # in order to fit the torch.fromnumpy
    
    
    if len(data.shape) == 2: # this is one band
        return geo, proj, data.reshape(data.shape[0], data.shape[1], 1)
    else:
        return geo, proj, data.transpose((1,2,0)) # in order to use the cv transform

def gdal_write(output_file,array_data, geo, proj):

  
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    h,w,c = array_data.shape
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file,w,h,c,datatype)
    
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)
    for i in range(c):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(array_data[:,:,i])


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="PREN", help="deep learning pan-sharpening method such as PNN MSDCNN PanNet TFNet SRPPNN and DIPNet")
parser.add_argument("-ms", "--model_sub", type=str, default="T6B6MSEB32", help="for our DIPNet's ablation study, such as SR SR_PLB SR_PHB SR_PLB_PHB default set as ''")
parser.add_argument("-ml", "--model_loss", type=str, default="", help="for our DIPNet's ablation study, such as SSIM or L1 or '', default set as ''")
parser.add_argument("-t", "--isTrain", type=bool, default=False, help="whether or not to train")
parser.add_argument("-g", "--gpu_ids", type=int, default=0, help="")
parser.add_argument("-c", "--continue_train", type=bool, default=False, help="whether or not to continue")
parser.add_argument("-we", "--which_epoch", type=int, default=-1, help="if continue Train ture, set this value")
parser.add_argument("-p", "--print_net_in_detail", type=bool, default=True, help="whether or not to continue")
parser.add_argument("-d", "--dataDir", type=str, default="", help="this is a train dataset dir")
parser.add_argument("-td", "--testdataDir", type=str, default="/home/sy/QB/dataset/test_dataset", help="this is a test dataset dir ")
parser.add_argument("-sr", "--save_result", type=bool, default=True, help="whether or not to save result")
parser.add_argument("-sd", "--saveDir", type=str, default="/home/sy/QB/result", help="save directory")
parser.add_argument("-cd", "--checkpoints_dir", type=str, default="/home/sy/QB/checkpoints", help="the dir of pertrained models or the dir to checkpoint the model parameters during training")
parser.add_argument("-nT", "--nThreads", type=int, default=0, help="use serval threads to load data in pytorch")
parser.add_argument("-bs", "--batchSize", type=int, default=16, help="")
parser.add_argument("-is", "--img_size", type=int, default=32, help="simulate ms size")
parser.add_argument("-sa", "--scale", type=int, default=4, help="scale factor which resizing to pan size")
parser.add_argument("-se", "--seed", type=int, default=19970716, help="random seed")
parser.add_argument("-pf", "--print_freq", type=int, default=5, help="print frequency of log")
parser.add_argument("-sf", "--save_epoch_freq", type=int, default=500, help="print frequency of log")
parser.add_argument("-pc", "--pan_channel", type=int, default=1, help="pan-chromatic band")
parser.add_argument("-mc", "--mul_channel", type=int, default=4, help="multi-spectral band which is based on different satellite")
parser.add_argument("-gm", "--gan_mode", type=str, default="lsgan", help="'lsgan' or 'wgangp' or 'vanilla' this is orginal")
parser.add_argument("-dr", "--data_range", type=int, default=2047, help="radis resolution ")
parser.add_argument("-lp", "--lr_policy", type=str, default='step', help="")
parser.add_argument("-ot", "--optim_type", type=str, default='adam', help="optim_type")
parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-b", "--beta", type=float, default=0.9, help="")
parser.add_argument("-mo", "--momentum", type=float, default=0.9, help="")
parser.add_argument("-w", "--weight_decay", type=float, default=1e-8, help="")
parser.add_argument("-li", "--lr_decay_iters", type=list, default=[1000], help="")
parser.add_argument("-lf", "--lr_decay_factor", type=float, default=0.5, help="")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="")
parser.add_argument("-iu", "--isUnlabel", type=bool, default=False, help="")
parser.add_argument("-ie", "--isEval", type=bool, default=False, help="")
parser.add_argument("-ia", "--isAug", type=bool, default=True, help="")
parser.add_argument("-uf", "--useFakePAN", type=bool, default=False, help="")
parser.add_argument("-sensor", "--sensor", type=str, default="", help="")
args = parser.parse_args()
input_data_path = args.testdataDir
input_ms_path = glob.glob(os.path.join(os.path.join(input_data_path, 'MS'), '*.tif'))
input_pan_path = glob.glob(os.path.join(os.path.join(input_data_path, 'PAN'), '*.tif'))
print(input_ms_path, input_pan_path)
input_ms_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'LR'), '*.tif'))
input_pan_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'HR'), '*.tif'))


if args.save_result:
  

    ours_path = os.path.join(args.saveDir, args.model+args.model_sub+args.model_loss)
    

    # dl supervised
    
    supervised_path = os.path.join(ours_path, 'supervised')
    
    
    reduce_path = os.path.join(supervised_path, 'reduce')
    full_path = os.path.join(supervised_path, 'full')
    
   
    if not os.path.exists(ours_path):
        os.makedirs(ours_path)
    

    
    if not os.path.exists(supervised_path):
        os.makedirs(supervised_path)
  

    
    
    if not os.path.exists(reduce_path):
        os.makedirs(reduce_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    
    # dl unsupervised
   
    ours_path = os.path.join(args.saveDir, args.model)
 
   
    unsupervised_path = os.path.join(ours_path, 'unsupervised')
   

   
    fake_path = os.path.join(unsupervised_path, 'fake')
    unfull_path = os.path.join(unsupervised_path, 'full')
    unreduce_path = os.path.join(unsupervised_path, 'reduce')

    if not os.path.exists(unsupervised_path):
        os.makedirs(unsupervised_path)


    if not os.path.exists(fake_path):
        os.makedirs(fake_path)
    if not os.path.exists(unfull_path):
        os.makedirs(unfull_path)
    if not os.path.exists(unreduce_path):
        os.makedirs(unreduce_path)


if args.model == 'ours':
    model = OursModel()

elif args.model == 'PNN':
    model = PNNModel()

elif args.model == 'PanNet':
    model = PanNetModel()
    
elif args.model == 'TFNet':
    model = TFNetModel()
    
elif args.model == 'MSDCNN':
    model = MSDCNNModel()
   
elif args.model == 'SRPPNN':
    model = SRPPNNModel()

elif args.model == 'PDIN':
    model = PDINModel()

elif args.model == 'DCNN':
    model = DCNNModel()

elif args.model == 'PDINSAM':
    model = PDINSAMModel()

elif args.model == 'PDINFS':
    model = PDINFSModel()

elif args.model == 'PDIMN':
    model = PDIMNModel()

elif args.model == 'HFNB6F64': 
    model = HFNModel()
    args.B = config_hfnb6f64.B
    args.features = config_hfnb6f64.features
    # args.B = 3
    # args.U = 2


    


elif args.model == 'HFN':  
    from model.HFN import HFNModel
    model = HFNModel()
    args.B = config_hfn.B
    args.features = config_hfn.features
    # args.B = 3
    # args.U = 2

elif args.model == 'TSDJN':  
    from model.TSDJN import TSDJNetModel
    model = TSDJNetModel()
    # args.B = 3
    # args.U = 2

elif args.model == 'HFNBaseline':  
    from model.HFNBaseline import HFNBaselineModel
    model = HFNBaselineModel()
    import options.config_hfnbaseline as hfnbaseline_cfg
    args.B = hfnbaseline_cfg.B
    args.features = hfnbaseline_cfg.features
    # args.B = 3
    # args.U = 2

elif args.model == 'HFNCA':  
    from model.HFNCA import HFNCAModel
    model = HFNCAModel()
    import options.config_hfnca as hfnca_cfg
    args.B = hfnca_cfg.B
    args.features = hfnca_cfg.features

elif args.model == 'HFNSA':  
    from model.HFNSA import HFNSAModel
    model = HFNSAModel()
    import options.config_hfnsa as hfnsa_cfg
    args.B = hfnsa_cfg.B
    args.features = hfnsa_cfg.features

elif args.model == 'HFNLSTM': 
    from model.HFNLSTM import HFNLSTMModel
    model = HFNLSTMModel()
    import options.config_hfnlstm as hfnlstm_cfg
    args.B = hfnlstm_cfg.B
    args.features = hfnlstm_cfg.features
    
elif args.model == 'PREN':  
    model = PReModel()
    args.T = pren_cfg.T

elif args.model == 'SSPN':
    model = SSPNModel()



model.initialize(args)
if args.scale == 6:
    input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
    input2 = torch.randn(1, args.pan_channel, 384, 384).cuda()
else:
    input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
    input2 = torch.randn(1, args.pan_channel, 256, 256).cuda()
flop, para = profile(model.netG, inputs=(input1, input2, ))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))

print('Total params: %.2fM' % (sum(p.numel() for p in model.netG.parameters())/1000000.0))
model.load_networks(999)

full_times = []
reduce_times = []

ergass = []
sams = []
sccs = []
qs = []
qaves = []
dss = []
d_ls = []
qnrs = []

for i, ms_path in tqdm(enumerate(input_ms_path)):
    # print(i)
    step = i
    
    '''loading data'''
    _, _, used_ms = gdal_read(ms_path)
    geo, proj, used_pan = gdal_read(input_pan_path[i])
    _, _, used_ms_down = gdal_read(input_ms_down_path[i])
    geo_down, proj_down, used_pan_down = gdal_read(input_pan_down_path[i])

    '''normalization'''
    used_ms = used_ms / args.data_range
    used_pan = used_pan / args.data_range
    used_ms_down = used_ms_down / args.data_range
    used_pan_down = used_pan_down / args.data_range

    name = os.path.basename(ms_path)    
    
    if args.isUnlabel:
        # start = time.time()
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        # end = time.time()
        # reduce_times.append(end-start)
        if args.save_result:
            gdal_write(os.path.join(unfull_path,name), fused_image, geo)
            
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
        if args.save_result:
            gdal_write(os.path.join(unreduce_path,name), fused_down_image, geo_down)
    else:
        s_time = time.time()
        
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        
        
        e_time = time.time()
        full_times.append(e_time-s_time)
        # ds = D_s(fused_image, cv2.resize(used_ms, dsize=(0,0),fx=args.scale, fy=args.scale), used_pan, args)
        # d_l = D_lambda(fused_image,  cv2.resize(used_ms, dsize=(0,0),fx=args.scale, fy=args.scale))
        # qnr = QNR(ds, d_l)
        # dss.append(ds)
        # d_ls.append(d_l)
        # qnrs.append(qnr)
        if args.save_result:
            gdal_write(os.path.join(full_path,name), fused_image, geo, proj)

        s_time = time.time()
        
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
       
        e_time = time.time()
        reduce_times.append(e_time-s_time)
        # ergas = ERGAS(fused_down_image, used_ms)
        # sam = SAM(fused_down_image, used_ms)
        # scc = SCC(fused_down_image, used_ms)
        # q = Q(fused_down_image, used_ms)
        # qave = QAVE(fused_down_image, used_ms)
        # ergass.append(ergas)
        # sams.append(sam)
        # sccs.append(scc)
        # qs.append(q)
        # qaves.append(qave)

        if args.save_result:
           gdal_write(os.path.join(reduce_path,name), fused_down_image, geo_down, proj_down)

    print('save done')

# print(np.mean(times), np.std(times))
print(args.model)
full_times = np.array(full_times)
reduce_times = np.array(reduce_times)
# ergass = np.array(ergass)
# sams = np.array(sams)
# sccs = np.array(sccs)
# qs = np.array(qs)
# qaves = np.array(qaves)
# dss = np.array(dss)
# d_ls = np.array(d_ls)
# qnrs = np.array(qnrs)
# print(args.sensor)
# print(args.model+args.model_sub+args.model_loss)


    





