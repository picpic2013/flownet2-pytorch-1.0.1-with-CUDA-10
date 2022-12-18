import torch
import torch.utils.data as data
import torchvision

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize
import cv2

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class PKUClassDataset(data.Dataset):
    def __init__(self, args, is_cropped = False, root=None):
        self.args = args
        self.rootFolder = root
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.imgList = [
            'bamboo', 'building', 'crossing', 'fern', 'horns', 'teddy'
        ]

    def __len__(self):
        return len(self.imgList) * 2

    def __getitem__(self, index):
        index0 = index // 2
        index1 = index % 2

        path1 = os.path.join(self.rootFolder, '%s%02d.png' % (self.imgList[index0], 1))
        path2 = os.path.join(self.rootFolder, '%s%02d.png' % (self.imgList[index0], 2))

        if index1 == 1:
            path1, path2 = path2, path1

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        images = [img1, img2]
        image_size = img1.shape[:2]

        H, W, _ = img1.shape
        flow = np.zeros([H, W, 2])

        frame_size = img1.shape
        render_size = self.render_size
        if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0]%64) or (frame_size[1]%64):
            render_size[0] = ( (frame_size[0])//64 ) * 64
            render_size[1] = ( (frame_size[1])//64 ) * 64

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.flo')) )
        for i in range(len(flows)):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Inspired by https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        image_1, label_1 = super(ImageFolderWithPaths, self).__getitem__(index)
        image_2, label_2 = super(ImageFolderWithPaths, self).__getitem__(index + 1)
        images = [image_1, image_2]
        labels = [label_1, label_2]
        # the image file path
        paths = self.imgs[index][0], self.imgs[index + 1][0]
        return images, labels, paths

    def __len__(self):
        return super(ImageFolderWithPaths, self).__len__() - 1

# Example Usage for VVT_ImageFolderWithPaths for train frames
data_dir = "/data_hdd/fw_gan_vvt/train/train_frames"
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])
# dataset = ImageFolderWithPaths(data_dir, transform=transforms) # our custom dataset
dataset = None # our custom dataset

# dataloader = torch.utils.data.DataLoader(dataset)
dataloader = None

# iterate over data
# inputs, labels, paths = [], [], []
# for input, label, path in dataloader:
#     # use the above variables freely
#     print(input, label, path)
#     image1, image2 = input
#     label1, label2 = label
#     path1, path2 = path
'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
