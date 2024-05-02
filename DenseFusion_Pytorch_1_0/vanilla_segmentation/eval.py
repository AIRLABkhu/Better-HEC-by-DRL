import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

from data_controller2 import SegDataset
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/airlab/projects/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed', help="dataset root dir (''YCB_Video Dataset'')")
parser.add_argument('--batch_size', default=2, help="batch size") #3
parser.add_argument('--n_epochs', default=500, help="epochs to train") #600
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers') #10
parser.add_argument('--lr', default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
opt = parser.parse_args()

if __name__ == '__main__':
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    test_dataset = SegDataset(opt.dataset_root, '/home/airlab/projects/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed/data/17_pp_test/test.txt', False, 1) #SegDataset(opt.dataset_root, '../datasets/ycb/dataset_config/test_data_list.txt', False, 1000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(opt.workers))

    print(len(test_dataset))

    model = segnet()
    model = model.cuda()
    
    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = Loss()
    best_val_cost = np.Inf
    st_time = time.time()

    for epoch in range(1, opt.n_epochs):
        model.eval()
        test_time = 0
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        for j, data in enumerate(test_dataloader, 0):
            rgb = data
            rgb = Variable(rgb).cuda()
            semantic = model(rgb)
            test_time += 1
            logger.info('Test time {0} Batch {1} semantic {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_time, semantic))
            semantic = semantic.detach().cpu().numpy()
            np.save('/home/airlab/projects/DenseFusion-Pytorch-1.0/vanilla_segmentation/num/{0}'.format(epoch),semantic)