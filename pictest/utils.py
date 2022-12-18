import cv2
import torch
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import os

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

# def calculatePSNRAndRender(img1, img2, flow, idx):
#     '''
#     @param img1 3 x H x W
#     @param flow 2 x H x W
#     '''
#     _, imgH, imgW = img1.shape
#     pixel_coordinate = np.indices([imgH, imgW])
#     V, U = torch.from_numpy(pixel_coordinate)
#     UV_Grid = torch.stack([U, V]).permute(1, 2, 0)[None]

#     flow_torch = flow[None].permute(0, 2, 3, 1)

#     UV_Grid = UV_Grid - flow_torch

#     normalize_base = torch.tensor([imgW * 0.5, imgH * 0.5])[None, None, None]

#     UV_Grid = UV_Grid / normalize_base - 1

#     img1_trans = img1[None].float()
#     img2_hat = torch.nn.functional.grid_sample(img1_trans, UV_Grid)[0].permute(1, 2, 0).numpy()

#     mask = np.sum(img2_hat, axis=2) != 0
#     psnr_value = psnr(img2.permute(1, 2, 0)[mask].numpy(), img2_hat[mask])

#     tmp = np.zeros([imgH * 3, imgW, 3])
#     tmp[:imgH] = img2.permute(1, 2, 0).numpy()
#     tmp[imgH:imgH*2] = img2_hat
#     tmp[imgH*2:] = img1.permute(1, 2, 0).numpy()
#     cv2.imwrite('outputs/%05d-%.5f.jpg' % (idx, psnr_value), tmp)

# def calculatePSNRAndRender(img1, img2, flow, idx):
#     '''
#     @param img1 3 x H x W
#     @param flow 2 x H x W
#     '''
#     _, imgH, imgW = img1.shape
#     pixel_coordinate = np.indices([imgH, imgW])
#     V, U = torch.from_numpy(pixel_coordinate)
#     UV_Grid = torch.stack([U, V]).permute(1, 2, 0)[None]

#     flow_torch = flow[None].permute(0, 2, 3, 1)

#     UV_Grid = UV_Grid + flow_torch

#     normalize_base = torch.tensor([imgW * 0.5, imgH * 0.5])[None, None, None]

#     UV_Grid = UV_Grid / normalize_base - 1

#     img2_trans = img2[None].float()
#     img1_hat = torch.nn.functional.grid_sample(img2_trans, UV_Grid)[0].permute(1, 2, 0).numpy()

#     mask = np.sum(img1_hat, axis=2) != 0
#     psnr_value = psnr(img1.permute(1, 2, 0)[mask].numpy(), img1_hat[mask])

#     tmp = np.zeros([imgH * 3, imgW, 3])
#     tmp[:imgH] = img2.permute(1, 2, 0).numpy()
#     tmp[imgH:imgH*2] = img1_hat
#     tmp[imgH*2:] = img1.permute(1, 2, 0).numpy()
#     cv2.imwrite('outputs/%05d-%.5f.jpg' % (idx, psnr_value), tmp)

def calculatePSNRAndRender(img1, img2, flow, idx):
    '''
    @param img1 3 x H x W
    @param flow 2 x H x W
    '''

    _, imgH, imgW = img1.shape
    pixel_coordinate = np.indices([imgH, imgW])
    V, U = torch.from_numpy(pixel_coordinate)
    UV_Grid = torch.stack([U, V]).permute(1, 2, 0)[None]

    flow_torch = flow[None].permute(0, 2, 3, 1)

    UV_Grid = UV_Grid + flow_torch

    normalize_base = torch.tensor([imgW * 0.5, imgH * 0.5])[None, None, None]

    UV_Grid = UV_Grid / normalize_base - 1

    img2_trans = img2[None].float()
    img1_hat = torch.nn.functional.grid_sample(img2_trans, UV_Grid)[0].permute(1, 2, 0).numpy()

    mask = np.sum(img1_hat, axis=2) != 0
    psnr_value = psnr(img1.permute(1, 2, 0)[mask].numpy(), img1_hat[mask])
    ssim_value = ssim(img1.permute(1, 2, 0).numpy(), img1_hat, multichannel=True)

    print(f"{idx} : psnr={psnr_value} ssim={ssim_value}")

    basePath = os.path.join('outputs', 'results', str(idx))
    if not os.path.exists(basePath):
        os.mkdir(basePath)

    cv2.imwrite(os.path.join(basePath, 'img1.png'), img1.permute(1, 2, 0).numpy())
    cv2.imwrite(os.path.join(basePath, 'img2.png'), img2.permute(1, 2, 0).numpy())
    cv2.imwrite(os.path.join(basePath, 'img1_hat.png'), img1_hat)

    with open(os.path.join(basePath, 'info.txt'), 'w') as f:
        f.write(f"psnr={psnr_value}\nssim={ssim_value}")