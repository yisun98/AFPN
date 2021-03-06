model ='AFPN'#'PNN'#'PanNet'#'TFNet'##'SRPPNN'# # deep learning pan-sharpening method such as PNN MSDCNN PanNet TFNet SRPPNN and DIPNet
model_sub = ''# for our DIPNet's ablation study, such as SR SR_PLB SR_PHB SR_PLB_PHB default set as ''
model_loss = '' # for our DIPNet's ablation study, such as SSIM or L1 or '', default set as ''
isTrain = True #whether or not to train
gpu_ids = '0'
continue_train = False# whether or not to continue
which_epoch = -1  # if continue Train ture, set this value
print_net_in_detail = True
dataDir = '/home/sy/QB/dataset/train_dataset'# this is a train dataset dir
testdataDir = '/home/sy/QB/dataset/train_dataset/test_dataset' # this is a test dataset dir
save_result = True # whether or not to save result
saveDir = '/home/sy/QB/results'# save directory
checkpoints_dir = '/home/sy/QB/checkpoints'# the dir of pertrained models or the dir to checkpoint the model parameters during training
nThreads = 0 # use serval threads to load data in pytorch
batchSize = 16
img_size = 48 # simulate ms size, only use in isAug
scale = 4 # scale factor which resizing to pan size
seed = 19970716 # random seed
print_freq = 1 # print frequency of log
save_epoch_freq = 100
pan_channel = 1 # pan-chromatic band
mul_channel = 4 # multi-spectral band which is based on different satellite
gan_mode = 'lsgan' #'lsgan' or 'wgangp' or 'vanilla' this is orginal
data_range = 2047 # radis resolution
lr_policy = 'step' #
optim_type = 'adam'#'adam'# c
lr = 1e-4
beta = 0.9
momentum = 0.9
weight_decay = 1e-8
lr_decay_iters = [500]
lr_decay_factor = 0.1
epochs = 1000
isUnlabel = False #
isEval = True#False
isAug = True #
useFakePAN = False
