import os
import sys
import time
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--inference-folder', type=str, default='', help='path to the folder containing demo images', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
parser.add_argument('--subset', type=str, default='pitts30k_test', help='Subset of Pitts250k on which inference is to be performed', choices=['pitts30k_train','pitts30k_val','pitts30k_test','pitts250k_val','pitts250k_test'])
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
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

if args.subset == 'pitts30k_test':
    structFile = join(struct_dir, 'pitts30k_test.mat')
    dbStruct = parse_dbStruct(structFile)
    image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
    image_dict += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
    
if args.subset == 'pitts30k_val':
    structFile = join(struct_dir, 'pitts30k_val.mat')
    dbStruct = parse_dbStruct(structFile)
    image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
    image_dict += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
 
if args.subset == 'pitts30k_train':
    structFile = join(struct_dir, 'pitts30k_train.mat')
    dbStruct = parse_dbStruct(structFile)
    image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
    image_dict += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
    
if args.subset == 'pitts250k_val':
    structFile = join(struct_dir, 'pitts250k_val.mat')
    dbStruct = parse_dbStruct(structFile)
    image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
    image_dict += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
    
if args.subset == 'pitts250k_test':
    structFile = join(struct_dir, 'pitts250k_test.mat')
    dbStruct = parse_dbStruct(structFile)
    image_dict = [join(root_dir, dbIm) for dbIm in dbStruct.dbImage]
    image_dict += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

images = image_dict
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

os.makedirs(args.save_dir,exist_ok=True)
os.makedirs(os.path.join(args.save_dir,'color_mask'),exist_ok=True)
#os.makedirs(os.path.join(args.save_dir,'overlap_images'),exist_ok=True)
#os.makedirs(os.path.join(args.save_dir,'pred_mask'),exist_ok=True)
os.makedirs(os.path.join(args.save_dir,'semantic_labels'),exist_ok=True)

start_time = time.time()
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    color_name = 'color_mask_' + img_name.replace('.jpg','.png')
    label_name = 'label_' + img_name.replace('.jpg','.npy')

    # save semantic labels
    np.save(os.path.join(os.path.join(args.save_dir,'semantic_labels'),label_name), pred)

    # save colorized predictions
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.save(os.path.join(os.path.join(args.save_dir,'color_mask'), color_name))

end_time = time.time()

print('Results saved.')
print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))
