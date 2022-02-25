# --------------------------------------------------------
# Script
# Copyright (c) 2021 EDL Lab
#
# Written by Bo Zhang
# --------------------------------------------------------
# To convert the label-level annotation map to the color-level figure
# -----------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2

def create_pascal_label_colormap():

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
  
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

LABEL_NAMES = np.asarray(['background', 'person'])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

imgfile = "/remote-home/V100/zb/IntraDA-gh-pages/test/I_top_mosaic_09cm_area15_18.png"
pngfile = "/remote-home/V100/zb/IntraDA-gh-pages/test/top_mosaic_09cm_area15_18.png"
img = cv2.imread(imgfile, 1)
img = img[:,:,::-1]
seg_map = cv2.imread(pngfile, 0)
vis_segmentation(img, seg_map)

print('Done.')