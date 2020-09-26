import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='data/training')
parser.add_argument("--checkpoint", type=str, default='checkpoint.pkl')
parser.add_argument("--img_num", type=int, default=0)
parser.add_argument("--disp_range", type=int, default=128)
args = parser.parse_args()


def load_sample(img_num):
    left_image = Image.open(os.path.join(args.data, 'image_2/%06d_10.png' % (img_num)))
    right_image = Image.open(os.path.join(args.data, 'image_3/%06d_10.png' % (img_num)))
    left_image = np.array(left_image)
    right_image = np.array(right_image)
    left_image = 255 * transforms.ToTensor()(left_image)
    right_image = 255 * transforms.ToTensor()(right_image)

    return left_image, right_image


disp_range = args.disp_range
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(3).to(device)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

left_img, right_image = load_sample(args.img_num)
left_img = (left_img - left_img.mean())/(left_img.std())
right_img = (right_image - right_image.mean())/(right_image.std())
img_height = left_img.size(1)
img_width = left_img.size(2)

left_img = left_img.view(1, left_img.size(0), left_img.size(1), left_img.size(2)).to(device)
right_img = right_img.view(1, right_img.size(0), right_img.size(1), right_img.size(2)).to(device)

start_time = time.time()
left_features = model.predict(left_img)
right_features = model.predict(right_img)
print(time.time() - start_time)

unary_vol = torch.zeros([img_height, img_width, disp_range])
right_unary_vol = torch.zeros([img_height, img_width, disp_range])
start_pos = 0
end_pos = img_width - 1
start_time = time.time()
while start_pos <= end_pos:
    for loc_idx in range(0, disp_range):
        x_off = -loc_idx + 1
        if end_pos+x_off >= 1 and img_width >= start_pos+x_off:
            l = left_features[:, :, :, np.max([start_pos, -x_off+1]): np.min([end_pos, img_width-x_off])]
            r = right_features[:, :, :, np.max([1, x_off+start_pos]): np.min([img_width, end_pos+x_off])]

            p = torch.mul(l, r)
            q = torch.sum(p, 1)
            unary_vol[:, np.max([start_pos, -x_off+1]): np.min([end_pos, img_width-x_off]), loc_idx] = q.data.view(q.data.size(1), q.data.size(2))
            right_unary_vol[:, np.max([1, x_off+start_pos]): np.min([img_width, end_pos+x_off]),
                            loc_idx] = q.data.view(q.data.size(1), q.data.size(2))

    start_pos = end_pos + 1

print(time.time() - start_time)
_, pred_1 = torch.max(unary_vol, 2)
_, pred_2 = torch.max(right_unary_vol, 2)

pred_disp1 = pred_1.view(unary_vol.size(0), unary_vol.size(1))
pred_disp2 = pred_2.view(unary_vol.size(0), unary_vol.size(1))

total_time = (time.time() - start_time)
print("Inference took: %fs" % total_time)

image = transforms.ToPILImage()(np.uint8(pred_disp1))
image.save('output-%d.png' % args.img_num)

im_gray = cv2.imread("output-%d.png" % args.img_num, cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(255-cv2.convertScaleAbs(im_gray, alpha=2), cv2.COLORMAP_JET)
cv2.imwrite("output-%d-color.png" % args.img_num, im_color)
plt.imshow(pred_disp1, cmap='gray')
plt.show()
