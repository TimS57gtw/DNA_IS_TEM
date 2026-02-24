import time

import cv2
import numpy as np
from random import randint
import os
from tqdm import tqdm
import glob

set = r'D:\seifert\PycharmProjects\LiuDimer\SynthData\Set5IS_400'
tf2 = r'F:\Data\LiuDimer\VisualizePred_NonInv\input'
outf = r'F:\Data\LiuDimer\VisualizePred_NonInv\NIP'
os.makedirs(outf, exist_ok=True)

subset = 'train'
imgs = [os.path.join(set, 'images', subset, x) for x in os.listdir(os.path.join(set, 'images', subset))]
lbls = [os.path.join(set, 'labels', subset, x) for x in os.listdir(os.path.join(set, 'labels', subset))]
imgs = glob.glob(os.path.join(tf2, '*'))
# for im, lb in tqdm(zip(imgs, lbls), total=len(imgs)):
for im in imgs:

    # with open(lb, 'r') as f:
    #     labels = f.read().splitlines()
    img = cv2.imread(im)
    h, w = img.shape[:2]
#
#
    # for label in labels:
    #     class_id, *poly = label.split(' ')
#
    #     # Reshape function
    #     start = time.perf_counter()
    #     xs = []
    #     ys = []
    #     poly = [float(x) for x in poly]
    #     for i, elem in enumerate(poly):
    #         if i % 2 == 0:
    #             xs.append(elem)
    #         else:
    #             ys.append(elem)
#
    #     pts = []
    #     for x, y in zip(xs, ys):
    #         v = np.array([x, y])
    #         pts.append(v)
#
    #     # print(pts)
#
#
    #     # pts = sorted(pts, key=clockwiseangle_and_distance)
    #     # print(pts)
    #     polyres = []
    #     for p in pts:
    #         polyres.append(p[0])
    #         polyres.append(p[1])
#
    #     poly = polyres
    #     # print('Dur: ', time.perf_counter() - start)
#
#
#
#
#
    #     poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
    #     poly *= [w, h]  # Unscale
#
    #     # cv2.polylines(img, [poly.astype('int')], True, (randint(0, 255), randint(0, 255), randint(0, 255)),
    #     #               2)  # Draw Poly Lines
    #     # cv2.fillPoly(img, [poly.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area
#
    #     if max(img.shape[0], img.shape[1]) < 400:
    #         img = cv2.resize(img, (600, 600))
#
    #     # cv2.imshow('img with poly', img)
    #     # cv2.waitKey(0)
#
    imagem = cv2.bitwise_not(img)

        # cv2.polylines(imagem, [poly.astype('int')], True, (0, 0, 255),
        #               2)  # Draw Poly Lines

    cv2.imwrite(os.path.join(outf, os.path.basename(im)), imagem)