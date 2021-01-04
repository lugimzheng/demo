import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from skimage import io
import cv2
from pose.network import CoordRegressionNetwork
from pose.networks import *



def pose_estimate(image):

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if True else "cpu")

    # image processing
    h, w = image.shape[:2]
    output_size = [224, 224]
    im_scale = min(float(output_size[0]) / float(h), float(output_size[1]) / float(w))
    new_h = int(image.shape[0] * im_scale)
    new_w = int(image.shape[1] * im_scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left_pad = (output_size[1] - new_w) // 2
    right_pad = (output_size[1] - new_w) - left_pad
    top_pad = (output_size[0] - new_h) // 2
    bottom_pad = (output_size[0] - new_h) - top_pad
    mean = np.array([0.485, 0.456, 0.406])
    pad = ((top_pad, bottom_pad), (left_pad, right_pad))
    image = np.stack([np.pad(image[:, :, c], pad, mode='constant', constant_values=mean[c])
                      for c in range(3)], axis=2)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image[:, :, :3] = (image[:, :, :3] - mean) / std
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    image = image.unsqueeze(0)

    # model
    modelname = "resnet18"
    modelpath = "./pose/models/%s_224_adam_best.t7" % (modelname)
    inputsize = 224
    net = CoordRegressionNetwork(n_locations=16, backbone=modelname).to(device)

    with torch.no_grad():
        net.load_state_dict(torch.load(modelpath))
        net = net.eval()
        image = image.to(device)
        coords, heatmaps = net(image)

    # drawing pose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.squeeze()
    pose = coords.squeeze()
    pose = pose.data.cpu().numpy()
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * std + mean, 0.0, 1.0)
    img_width, img_height, _ = img.shape
    pose = ((pose + 1) * np.array([img_width, img_height]) - 1) / 2

    pose = (pose - np.array([left_pad, top_pad])) / im_scale
    colors = [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [85, 0, 255],
      [85, 0, 255], [0, 255, 170], [0, 255, 170], [0, 85, 255], [0, 85, 255], [0, 85, 255], [0, 85, 255],
      [0, 85, 255], [0, 85, 255]]

    pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
    colors_skeleton = [[0, 255, 170], [0, 85, 255], [0, 85, 255], [255, 0, 0], [255, 0, 0], [0, 85, 255], [0, 85, 255],
           [255, 0, 0], [255, 0, 0], [85, 0, 255], [85, 0, 255], [255, 0, 0], [255, 0, 0], [0, 85, 255],
           [0, 85, 255]]

    return pose, colors, pairs, colors_skeleton
