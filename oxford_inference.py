import os
import sys
import time
import argparse
from PIL import Image
import numpy as np
import cv2
from os.path import join, exists, isfile, realpath, dirname, basename
from os import makedirs, remove, chdir, environ
from collections import namedtuple
from scipy.io import loadmat

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

parser = argparse.ArgumentParser(description='Oxford Inference')
parser.add_argument('--inference-folder', type=str, default='', help='path to the folder containing demo images', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

# get data
root_dir = args.inference_folder
if not os.path.exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Oxford dataset')

dbstruct = namedtuple('Struct', [
    'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['dbStruct'][0]

    dataset = 'oxford'

    whichSet = 'NightVsDay'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr, 
            posDistSqThr)

structFile = join(root_dir, 'oxdatapart.mat')
dbStruct = parse_dbStruct(structFile)
image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
#image_dict += [join(root_dir, qIm) for qIm in dbStruct.qImage]
    
images = image_dict
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

os.makedirs(args.save_dir,exist_ok=True)
os.makedirs(os.path.join(args.save_dir,'1-s/color_mask'),exist_ok=True)
# os.makedirs(os.path.join(args.save_dir,'semantic_labels'),exist_ok=True)
os.makedirs(os.path.join(args.save_dir,'1-s/semantic_prob'),exist_ok=True)

start_time = time.time()
for img_id, img_dir in enumerate(images):
    #img_dir = os.path.join(data_dir, img_name)
    img_name = basename(img_dir)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

    pred = pred.cpu().numpy().squeeze()
    prob = pred
    pred = np.argmax(pred, axis=0)

    color_name = 'color_mask_' + img_name
    label_name = 'label_' + img_name
    prob_name = 'prob_' + img_name

    # save semantic labels
    #np.save(os.path.join(os.path.join(args.save_dir,'semantic_labels'),label_name), pred)
    np.save(os.path.join(os.path.join(args.save_dir,'1-s/semantic_prob'),prob_name), prob)

    # save colorized predictions
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.save(os.path.join(os.path.join(args.save_dir,'1-s/color_mask'), color_name))

end_time = time.time()

print('Results saved.')
print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))
