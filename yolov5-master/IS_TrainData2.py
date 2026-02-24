import copy
import os
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from perlin_noise import PerlinNoise
from tqdm import tqdm
from shapely.geometry.polygon import Polygon
from scipy.special import binom
from shapely.geometry import Point
import OrganicShapes
from multiprocessing import Process



RESOLUTION = 200


def analyse_image(fn):
    arr = np.array(Image.open(fn))[:, :, 0].astype(float)

    arr /= 255

    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(arr)
    plt.show()

    vals = arr.flatten()
    plt.hist(vals,bins=100)
    plt.show()

    arr_back = copy.deepcopy(arr)
    arr_back[100:300, 100:300] = 0
    vals = [x for x in arr_back.flatten() if x > 0]

    plt.hist(vals, bins=100)
    mu = np.mean(vals)
    sig = np.std(vals) / mu
    plt.title(f'mu: {mu:.3f} sig={sig:.3f}')
    plt.show()
    print(np.amin(arr), np.amax(arr))


    mid = arr[100:300, 100:300]
    plt.imshow(mid, cmap='gray')
    plt.title('mid')
    plt.show()


    return mu, sig


class Rect:
        def __init__(self, v1, v2, v3, v4) -> None:
            self.v1 = np.array(v1)
            self.v2 = np.array(v2)
            self.v3 = np.array(v3)
            self.v4 = np.array(v4)

            self.tri1 = ([self.v1[0], self.v1[1]], [self.v2[0], self.v2[1]], [self.v3[0], self.v3[1]])
            self.tri2 = ([self.v2[0], self.v2[1]], [self.v3[0], self.v3[1]], [self.v4[0], self.v4[1]])
            vec1 = self.v3 - self.v1
            vec2 = self.v2 - self.v1
            cp = np.cross(vec1, vec2)
            a, b, c = cp
            d = np.dot(cp, self.v3)
            self.plane1 = lambda x : (d -a*x[0] - b*x[1])/c
            vec12 = self.v3 - self.v4
            vec22 = self.v2 - self.v4
            cp2 = np.cross(vec12, vec22)
            a2, b2, c2 = cp2
            d2 = np.dot(cp2, self.v3)
            self.plane2 = lambda x : (d2 -a2*x[0] - b2*x[1])/c2

        def sign(self, p1, p2, p3):

            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
        def pointInTriangle(self, pt, tri):
            d1 = self.sign(pt, tri[0], tri[1]);
            d2 = self.sign(pt, tri[1], tri[2]);
            d3 = self.sign(pt, tri[2], tri[0]);
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0);
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0);
            return not (has_neg and has_pos);
        def contains(self, vec):
            return self.pointInTriangle(vec, self.tri1) or self.pointInTriangle(vec, self.tri2)
        def elevation(self, vec):
            vecs = []
            if self.pointInTriangle(vec, self.tri1):
                return self.plane1(vec)
            elif self.pointInTriangle(vec,self.tri2):
                return self.plane2(vec)
            else:
                print("Does not contain")
                return 100



        def __str__(self) -> str:
            return "Rect: ({:.1f},{:.1f},{:.3f}) - ({:.1f},{:.1f},{:.3f}) - ({:.1f},{:.1f},{:.3f}) - ({:.1f},{:.1f},{:.3f})".format(self.v1[0],self.v1[1],self.v1[2],
                                                                                                                                        self.v2[0],self.v2[1],self.v2[2],
                                                                                                                                        self.v3[0],self.v3[1],self.v3[2],
                                                                                                                                        self.v4[0],self.v4[1],self.v4[2],)



def gen_DNA_shape():
    arr = np.zeros((RESOLUTION, RESOLUTION))

    # pts1 = np.array([np.random.normal(RESOLUTION/4, RESOLUTION/8), np.random.normal(RESOLUTION/4, RESOLUTION/8)])
    # pts2 = np.array([np.random.normal(RESOLUTION/4, RESOLUTION/8), np.random.normal(3*RESOLUTION/4, RESOLUTION/8)])
    # pts3 = np.array([np.random.normal(3*RESOLUTION/4, RESOLUTION/8), np.random.normal(3*RESOLUTION/4, RESOLUTION/8)])
    # pts4 = np.array([np.random.normal(3*RESOLUTION/4, RESOLUTION/8), np.random.normal(RESOLUTION/4, RESOLUTION/8)])
#
    # pol = Polygon([pts1, pts2, pts3, pts4])

    scale = max(0.1, np.random.normal(0.3, 0.1)) * RESOLUTION
    npts = 4
    rad = np.random.uniform(0, 0.3)
    edgy = np.random.uniform(0.1, 0.5)




    scl = np.random.uniform(scale[0], scale[1]) if type(scale) is tuple else scale
    npl = np.random.randint(npts[0], npts[1]) if type(npts) is tuple else npts

    sig= scl / 10
    theta = 2*np.pi * np.random.random()
    def get_random_pts(scale):
        ar = np.random.uniform(0.4, 1)
        wd = np.sqrt(scale**2 / ar)
        hd = wd * ar
        tl = np.array([np.random.normal(-wd/2, sig), np.random.normal(-hd/2, sig)])
        bl = np.array([np.random.normal(wd/2, sig), np.random.normal(-hd/2, sig)])
        br = np.array([np.random.normal(wd/2, sig), np.random.normal(hd/2, sig)])
        tr = np.array([np.random.normal(-wd/2, sig), np.random.normal(hd/2, sig)])

        ml = np.array([np.random.normal(0, sig), min(np.random.normal(-hd/2, sig), (tl[1] + bl[1]) / 2)])
        mr = np.array([np.random.normal(0, sig), max(np.random.normal(hd/2, sig), (tr[1] + br[1]) / 2)])
        tm = np.array([min(np.random.normal(-wd/2, sig), (tl[0] + tr[0])/2), np.random.normal(0, sig)])
        bm = np.array([max(np.random.normal(wd/2, sig), (bl[0] + br[0])/2), np.random.normal(0, sig)])


        R = np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])
        tl = np.dot(tl, R)
        bl = np.dot(bl, R)
        br = np.dot(br, R)
        tr = np.dot(tr, R)
        ml = np.dot(ml, R)
        mr = np.dot(mr, R)
        bm = np.dot(bm, R)
        tm = np.dot(tm, R)

        # tl += np.array([0.5, 0.5])
        # bl += np.array([0.5, 0.5])
        # br += np.array([0.5, 0.5])
        # tr += np.array([0.5, 0.5])

        # pts = np.zeros((8, 2))
        # pts[0, :] = tl
        # pts[1, :] = bl
        # pts[2, :] = br
        # pts[3, :] = tr
        # pts[4, :] = ml
        # pts[5, :] = mr
        # pts[6, :] = bm
        # pts[7, :] = tm

        pts = np.zeros((8, 2))
        pts[0, :] = tl
        pts[1, :] = ml
        pts[2, :] = bl
        pts[3, :] = bm
        pts[4, :] = br
        pts[5, :] = mr
        pts[6, :] = tr
        pts[7, :] = tm


        return pts





    #a = OrganicShapes.get_random_points(n=npl, scale=scl)

    a = get_random_pts(scale=scl)





    x, y, _ = OrganicShapes.get_bezier_curve(a, rad=rad, edgy=edgy)
    xmid = np.random.uniform(RESOLUTION/4, 3*RESOLUTION/4)
    ymid = np.random.uniform(RESOLUTION/4, 3*RESOLUTION/4)

    avx = np.average(x)
    avy = np.average(y)

    x += (xmid -avx)
    y += (ymid -avy)

    lsx = x[-1]
    lsy = y[-1]
    x = x[::10]
    y = y[::10]
    x[-1] = lsx
    y[-1] = lsy

   #  print(x)
   #  print(y)

    xs_lbl = []
    ys_lbl = []



    for i in range(len(a)):
        ys_lbl.append(a[i][0] + xmid - avx)
        xs_lbl.append(a[i][1] + ymid - avy)

    # print('\nx', xs_lbl)
    # print('y', ys_lbl)



    pol = Polygon(zip(x, y))


    for i in tqdm(range(RESOLUTION), disable=True):
        for j in range(RESOLUTION):
            for p in [pol]:
                pt = Point(i, j)
                if p.contains(pt):
                    arr[i, j] = 1
                    break


   #  plt.imshow(arr)
   #  plt.show()

    # plt.imshow(arr)
    # plt.show()

    lbl = copy.deepcopy(arr)

    ks = int(RESOLUTION / 20) * 2 + 1
    mid = int(np.ceil(ks/2))
    kernelsig = np.random.uniform(5 * ks / 40, 30 * ks / 40)
    kernel = np.zeros((ks, ks))
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / kernelsig)

    kernel /= np.sum(kernel)
   #  plt.imshow(kernel)
   #  plt.title('Large Kernel')
   #  plt.show()

    arr = arr.astype(np.uint8)
    arr *= 255
    arr = cv2.filter2D(arr.astype(np.uint8), -1, kernel)
    arr = arr.astype(float)
    arr /= 255

    # plt.imshow(arr)
    # plt.show()

    lbl_txt = '0'

    for i in range(len(xs_lbl)):
        lbl_txt += ' '
        lbl_txt += str(xs_lbl[i] / arr.shape[0])
        lbl_txt += ' '
        lbl_txt += str(ys_lbl[i] / arr.shape[1])

    lbl_txt += '\n'

    return arr, lbl, lbl_txt





def gen_DNA_shape_many():
    arr = np.zeros((RESY, RESX))

    # pts1 = np.array([np.random.normal(RESOLUTION/4, RESOLUTION/8), np.random.normal(RESOLUTION/4, RESOLUTION/8)])
    # pts2 = np.array([np.random.normal(RESOLUTION/4, RESOLUTION/8), np.random.normal(3*RESOLUTION/4, RESOLUTION/8)])
    # pts3 = np.array([np.random.normal(3*RESOLUTION/4, RESOLUTION/8), np.random.normal(3*RESOLUTION/4, RESOLUTION/8)])
    # pts4 = np.array([np.random.normal(3*RESOLUTION/4, RESOLUTION/8), np.random.normal(RESOLUTION/4, RESOLUTION/8)])
#
    # pol = Polygon([pts1, pts2, pts3, pts4])


    no_mols = np.random.randint(5, 30)
    midpoints = []

    for i in tqdm(range(no_mols), desc='genMol'):

        scale = max(0.01, np.random.normal(0.03, 0.01)) * np.sqrt(RESX**2 + RESY**2)



        npts = 4
        rad = np.random.uniform(0, 0.3)
        edgy = np.random.uniform(0.1, 0.5)




        scl = np.random.uniform(scale[0], scale[1]) if type(scale) is tuple else scale
        npl = np.random.randint(npts[0], npts[1]) if type(npts) is tuple else npts

        sig= scl / 10
        theta = 2*np.pi * np.random.random()
        def get_random_pts(scale):
            ar = np.random.uniform(0.4, 1)
            wd = np.sqrt(scale**2 / ar)
            hd = wd * ar
            tl = np.array([np.random.normal(-wd/2, sig), np.random.normal(-hd/2, sig)])
            bl = np.array([np.random.normal(wd/2, sig), np.random.normal(-hd/2, sig)])
            br = np.array([np.random.normal(wd/2, sig), np.random.normal(hd/2, sig)])
            tr = np.array([np.random.normal(-wd/2, sig), np.random.normal(hd/2, sig)])

            ml = np.array([np.random.normal(0, sig), min(np.random.normal(-hd/2, sig), (tl[1] + bl[1]) / 2)])
            mr = np.array([np.random.normal(0, sig), max(np.random.normal(hd/2, sig), (tr[1] + br[1]) / 2)])
            tm = np.array([min(np.random.normal(-wd/2, sig), (tl[0] + tr[0])/2), np.random.normal(0, sig)])
            bm = np.array([max(np.random.normal(wd/2, sig), (bl[0] + br[0])/2), np.random.normal(0, sig)])


            R = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
            tl = np.dot(tl, R)
            bl = np.dot(bl, R)
            br = np.dot(br, R)
            tr = np.dot(tr, R)
            ml = np.dot(ml, R)
            mr = np.dot(mr, R)
            bm = np.dot(bm, R)
            tm = np.dot(tm, R)

            # tl += np.array([0.5, 0.5])
            # bl += np.array([0.5, 0.5])
            # br += np.array([0.5, 0.5])
            # tr += np.array([0.5, 0.5])

            # pts = np.zeros((8, 2))
            # pts[0, :] = tl
            # pts[1, :] = bl
            # pts[2, :] = br
            # pts[3, :] = tr
            # pts[4, :] = ml
            # pts[5, :] = mr
            # pts[6, :] = bm
            # pts[7, :] = tm

            pts = np.zeros((8, 2))
            pts[0, :] = tl
            pts[1, :] = ml
            pts[2, :] = bl
            pts[3, :] = bm
            pts[4, :] = br
            pts[5, :] = mr
            pts[6, :] = tr
            pts[7, :] = tm


            return pts





        #a = OrganicShapes.get_random_points(n=npl, scale=scl)

        a = get_random_pts(scale=scl)





        x, y, _ = OrganicShapes.get_bezier_curve(a, rad=rad, edgy=edgy)

        redo = False
        for _ in tqdm(range(100), desc='Find MP', disable=True):
            redo = False
            ymid = np.random.uniform(0, RESX)
            xmid = np.random.uniform(0, RESY)
            pt = np.array([xmid, ymid])
            for op in midpoints:
                if np.linalg.norm(pt - op) < 7 * scale:
                    redo = True
                    break

            if not redo:
                break

        if redo:
            continue

        midpoints.append(pt)

        avx = np.average(x)
        avy = np.average(y)

        x += (xmid -avx)
        y += (ymid -avy)

        lsx = x[-1]
        lsy = y[-1]
        x = x[::10]
        y = y[::10]
        x[-1] = lsx
        y[-1] = lsy

       #  print(x)
       #  print(y)

        xs_lbl = []
        ys_lbl = []



        for i in range(len(a)):
            ys_lbl.append(a[i][0] + xmid - avx)
            xs_lbl.append(a[i][1] + ymid - avy)

        # print('\nx', xs_lbl)
        # print('y', ys_lbl)



        pol = Polygon(zip(x, y))


        for i in tqdm(range(max(0, int(ymid - 200)), min(int(ymid + 200), arr.shape[0]-1)), disable=True, desc='GenLabel'):
            for j in range(max(0, int(xmid - 200)), min(arr.shape[1] - 1, int(xmid + 200))):
                for p in [pol]:
                    pt = Point(i, j)
                    if p.contains(pt):
                        arr[i, j] = 1
                        break


   #  plt.imshow(arr)
   #  plt.show()

    # plt.imshow(arr)
    # plt.show()

    lbl = copy.deepcopy(arr)

    ks = int(max(RESX, RESY) / 20) * 2 + 1
    mid = int(np.ceil(ks/2))
    kernelsig = np.random.uniform(5 * ks / 40, 30 * ks / 40)
    kernel = np.zeros((ks, ks))
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / kernelsig)

    kernel /= np.sum(kernel)
   #  plt.imshow(kernel)
   #  plt.title('Large Kernel')
   #  plt.show()

    arr = arr.astype(np.uint8)
    arr *= 255
    arr = cv2.filter2D(arr.astype(np.uint8), -1, kernel)
    arr = arr.astype(float)
    arr /= 255

    # plt.imshow(arr)
    # plt.show()

    lbl_txt = '0'

    for i in range(len(xs_lbl)):
        lbl_txt += ' '
        lbl_txt += str(xs_lbl[i] / arr.shape[0])
        lbl_txt += ' '
        lbl_txt += str(ys_lbl[i] / arr.shape[1])

    lbl_txt += '\n'

    return arr, lbl, lbl_txt






def generate_image(save_fn, label_fn, label_txt_fn):
    arr = np.zeros((RESOLUTION, RESOLUTION))

    octaves = np.random.randint(12, 28)

    noise_mu = np.abs(np.random.normal(0.5, 0.104))
    noise_mu = np.clip(noise_mu, 0.05, 0.75)
    noise_sig = 0.05 + 0.1 * np.random.random()
    sig = noise_sig * noise_mu
    sig = min(0.5, sig)


    perlin_x = PerlinNoise(octaves=octaves)
    noise = [[perlin_x([i / RESOLUTION, j / RESOLUTION]) for j in range(RESOLUTION)] for i in
                    range(RESOLUTION)]

    noise -= np.average(noise)
    noise /= np.std(noise)
    noise *= sig
    noise += noise_mu

    # print(f"Mu: Soll={noise_mu}, Ist{np.average(noise)}, diff: {100 * (noise_mu - np.average(noise)) / noise_mu :.2f}%")
    # print(f"Sig: Soll={noise_sig}, Ist{np.std(noise) / np.average(noise)}, diff: {100 * (noise_sig - (np.std(noise) / np.average(noise))) / noise_sig :.2f}%")


    # plt.imshow(noise, cmap='gray', vmin=0, vmax=1)
    # plt.title('Noise')
    # plt.show()



    mol_height = np.random.uniform(0.1, max(0.2, 0.9-noise_mu))

    mol_arr, label, label_text = gen_DNA_shape()

    with open(label_txt_fn, 'w') as f:
        f.write(label_text)


    mol_arr *= mol_height

    white_noise = np.random.normal(0.1, 0.05, arr.shape)


    no_spots = np.random.randint(0, 10)
    diam_x = np.random.normal(100 * RESOLUTION / 400, 20 * RESOLUTION / 400, no_spots)
    diam_y = np.random.normal(100 * RESOLUTION / 400, 20 * RESOLUTION / 400, no_spots)
    heights = []
    for i in range(no_spots):
        if random.random() < 2:
            heights.append( - np.random.uniform((mol_height + noise_mu)/2, mol_height + noise_mu))
        else:
            heights.append(np.random.random())

    spot_arr = np.zeros_like(arr)

    # calc mol center
    non_zero_indices = np.nonzero(label)
    average_position = np.mean(np.column_stack(non_zero_indices), axis=0)

    # plt.imshow(label)
    # plt.show()


    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    borderT = cv2.filter2D(label.astype(np.uint8), -1, kernel)

    k2s = int((25 * RESOLUTION / 400) / 2) * 2 + 1
    kernel2 = np.ones((k2s, k2s))
    border = cv2.filter2D(borderT.astype(np.uint8), -1, kernel2)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(label)
    # axs[1].imshow(borderT)
    # axs[2].imshow(border)
#
    # plt.show()

    brd = np.argwhere(border > 0)



    for i in range(no_spots):
        ordered = np.random.random() < 0.7
        border = np.random.random() < 0.5
        if ordered:
            if border:
                pos = brd[np.random.randint(0, len(brd))]
                posx = pos[0]
                posy = pos[1]

            else:
                posy = np.random.normal(average_position[1], RESOLUTION / 5)
                posx = np.random.normal(average_position[0], RESOLUTION / 5)
        else:
            posx = np.random.randint(0, RESOLUTION)
            posy = np.random.randint(0, RESOLUTION)
        sigma_X = np.sqrt(diam_x[i])
        sigma_Y = np.sqrt(diam_y[i])
        rng_x = int(3*sigma_X)
        rng_y = int(3*sigma_Y)
        height = heights[i]
        theta = np.random.random() * 2 * np.pi
        a = np.cos(theta) ** 2 / (2 * sigma_X ** 2) + np.sin(theta) ** 2 / (2 * sigma_Y ** 2)
        b = np.sin(2 * theta) / (4 * sigma_X ** 2) - np.sin(2 * theta) / (4 * sigma_Y ** 2)
        c = np.sin(theta) ** 2 / (2 * sigma_X ** 2) + np.cos(theta) ** 2 / (2 * sigma_Y ** 2)

        if random.random() > 0.5:
            hfkt = lambda x : np.exp(-x)
        else:
            fk = np.random.uniform(0.5, 3)
            hfkt = lambda x : np.cos(fk * x) * np.exp(-x)

        for i in range(int(posx) - rng_x, int(posx) + rng_x):
            for j in range(int(posy) - rng_y, int(posy) + rng_y):
                h = height * hfkt((a * (i - posx)**2 + 2 * b * (i - posx) * (j - posy) + c * (j - posy)**2))
                # h = height * np.exp(-(a * (i - posx)**2 + 2 * b * (i - posx) * (j - posy) + c * (j - posy)**2))
                if 0 <= i < spot_arr.shape[0] and 0 <= j < spot_arr.shape[1]:
                    spot_arr[i, j] += h

        # plt.imshow(spot_arr)
        # plt.title(f'Spots {posx}, {posy}, {sigma_X}, {sigma_Y}')
        # plt.show()
    spal = copy.deepcopy(spot_arr)
    spal = np.abs(spal)
    kernel3 = np.ones((7,7))
    spal = cv2.filter2D(spal, -1, kernel3)

    spt_th = 2
    spLbl = np.zeros_like(spal)
    spLbl[spal > spt_th] = 2

    # fig, axs = plt.subplots(2)
    # axs[0].imshow(spot_arr)
    # axs[1].imshow(spLbl)
    # plt.show()

    arr = noise + mol_arr + white_noise + spot_arr
    arr = np.clip(arr, 0, 1)

    label += spLbl



    # plt.imshow(arr, vmin=0, vmax=1, cmap='gray')
    # plt.show()

    plt.imsave(save_fn, arr, vmin=0, vmax=1, cmap='gray')
    plt.imsave(label_fn, label, vmin=0, vmax=2, cmap='gray')


def generate_image_large(save_fn, label_fn, label_txt_fn):
    arr = np.zeros((RESY, RESX))

    octaves = np.random.randint(20, 40)

    noise_mu = np.abs(np.random.normal(0.33, 0.104))
    noise_mu = np.clip(noise_mu, 0.05, 0.85)
    noise_sig = 0.05 + 0.1 * np.random.random()
    sig = noise_sig * noise_mu
    sig = min(0.5, sig)


    perlin_x = PerlinNoise(octaves=octaves)
    mr = max(RESX, RESY)
    noise = [[perlin_x([i / mr, j / mr]) for j in range(RESY)] for i in
                    range(RESX)]

    noise -= np.average(noise)
    noise /= np.std(noise)
    noise *= sig
    noise += noise_mu

    # print(f"Mu: Soll={noise_mu}, Ist{np.average(noise)}, diff: {100 * (noise_mu - np.average(noise)) / noise_mu :.2f}%")
    # print(f"Sig: Soll={noise_sig}, Ist{np.std(noise) / np.average(noise)}, diff: {100 * (noise_sig - (np.std(noise) / np.average(noise))) / noise_sig :.2f}%")


    # plt.imshow(noise, cmap='gray', vmin=0, vmax=1)
    # plt.title('Noise')
    # plt.show()


    mol_height = np.random.uniform(0.1, max(0.2, 0.9-noise_mu))

    mol_arr, label, label_text = gen_DNA_shape_many()

    with open(label_txt_fn, 'a') as f:
        f.write(label_text)


    mol_arr *= mol_height

    white_noise = np.random.normal(0.1, 0.05, arr.shape)


    no_spots = np.random.randint(0, 20)
    res = np.sqrt(RESX**2 + RESY**2)
    diam_x = np.random.normal(100 * res / 400, 20 * res / 400, no_spots)
    diam_y = np.random.normal(100 * res / 400, 20 * res / 400, no_spots)
    heights = []
    for i in range(no_spots):
        if random.random() < 2:
            heights.append( - np.random.uniform((mol_height + noise_mu)/2, mol_height + noise_mu))
        else:
            heights.append(np.random.random())

    spot_arr = np.zeros_like(arr)

    # calc mol center
    non_zero_indices = np.nonzero(label)
    average_position = np.mean(np.column_stack(non_zero_indices), axis=0)

    # plt.imshow(label)
    # plt.show()


    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    borderT = cv2.filter2D(label.astype(np.uint8), -1, kernel)

    k2s = int((25 * RESOLUTION / 400) / 2) * 2 + 1
    kernel2 = np.ones((k2s, k2s))
    border = cv2.filter2D(borderT.astype(np.uint8), -1, kernel2)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(label)
    # axs[1].imshow(borderT)
    # axs[2].imshow(border)
#
    # plt.show()

    brd = np.argwhere(border > 0)



    for i in range(no_spots):
        ordered = np.random.random() < 0.7
        border = np.random.random() < 0.5
        if ordered:
            if border and len(brd) > 0:
                pos = brd[np.random.randint(0, len(brd))]
                posx = pos[0]
                posy = pos[1]

            else:
                posy = np.random.normal(average_position[1], RESOLUTION / 5)
                posx = np.random.normal(average_position[0], RESOLUTION / 5)
        else:
            posx = np.random.randint(0, RESX)
            posy = np.random.randint(0, RESY)
        sigma_X = np.sqrt(diam_x[i])
        sigma_Y = np.sqrt(diam_y[i])
        rng_x = int(3*sigma_X)
        rng_y = int(3*sigma_Y)
        height = heights[i]
        theta = np.random.random() * 2 * np.pi
        a = np.cos(theta) ** 2 / (2 * sigma_X ** 2) + np.sin(theta) ** 2 / (2 * sigma_Y ** 2)
        b = np.sin(2 * theta) / (4 * sigma_X ** 2) - np.sin(2 * theta) / (4 * sigma_Y ** 2)
        c = np.sin(theta) ** 2 / (2 * sigma_X ** 2) + np.cos(theta) ** 2 / (2 * sigma_Y ** 2)

        if random.random() > 0.5:
            hfkt = lambda x : np.exp(-x)
        else:
            fk = np.random.uniform(0.5, 3)
            hfkt = lambda x : np.cos(fk * x) * np.exp(-x)

        for i in range(max(0, int(posx - rng_x)), min(spot_arr.shape[0] - 1, int(int(posx) + rng_x))):
            for j in range(max(int(posy - rng_y), 0), min(spot_arr.shape[1] - 1, int(posy) + rng_y)):
                h = height * hfkt((a * (i - posx)**2 + 2 * b * (i - posx) * (j - posy) + c * (j - posy)**2))
                # h = height * np.exp(-(a * (i - posx)**2 + 2 * b * (i - posx) * (j - posy) + c * (j - posy)**2))
                if 0 <= i < spot_arr.shape[0] and 0 <= j < spot_arr.shape[1]:
                    spot_arr[i, j] += h

        # plt.imshow(spot_arr)
        # plt.title(f'Spots {posx}, {posy}, {sigma_X}, {sigma_Y}')
        # plt.show()
    spal = copy.deepcopy(spot_arr)
    spal = np.abs(spal)
    kernel3 = np.ones((7,7))
    spal = cv2.filter2D(spal, -1, kernel3)

    spt_th = 2
    spLbl = np.zeros_like(spal)
    spLbl[spal > spt_th] = 2

    # fig, axs = plt.subplots(2)
    # axs[0].imshow(spot_arr)
    # axs[1].imshow(spLbl)
    # plt.show()


    arr = noise.T + mol_arr + white_noise + spot_arr
    arr = np.clip(arr, 0, 1)

    label += spLbl



    # plt.imshow(arr, vmin=0, vmax=1, cmap='gray')
    # plt.show()

    plt.imsave(save_fn, arr, vmin=0, vmax=1, cmap='gray')
    plt.imsave(label_fn, label, vmin=0, vmax=2, cmap='gray')


class gen_parallel(Process):
    def __init__(self, fnames, resf, lblf, isf, tqd):
        super().__init__()

        self.fnames = fnames
        self.resf = resf
        self.lblf = lblf
        self.tqd = tqd
        self.isf = isf

    def run(self) -> None:
        for fn in tqdm(self.fnames, disable=not self.tqd):
            imf = os.path.join(self.resf, fn)
            lblf = os.path.join(self.lblf, fn)
            lblf_IS = os.path.join(self.isf, fn.split('.')[0] + '.txt')
            generate_image(imf, lblf, lblf_IS)


RESX = 3072
RESY = 2048

# RESX = 1000
# RESY = 500
if __name__ == '__main__':
    idx = 0
    # ldf = r'D:\seifert\PycharmProjects\LiuDimer\RealCrops'
    ct = 0
   #  if False:
   #      fns = list(os.listdir(ldf))
   #      np.random.shuffle(fns)
   #      with open('stats.csv', 'w') as f:
   #          f.write('mu;sig\n')
   #          for fn in tqdm(fns):
   #              ct += 1
   #              fn = random.choice(os.listdir(ldf))
   #              mu, sig = analyse_image(os.path.join(ldf, fn))
   #              f.write(f'{mu};{sig}\n')
   #              if ct > 200:
   #                  break
#
#
   #  while False:
        # gen_DNA_shape()

    num = 100
    threads = 10

    fn_lists = []
    for i in range(threads):
        fn_lists.append([])

    for i in range(num):
        fn_lists[i%threads].append(f'{str(i).zfill(6)}.png')

    outfld = os.path.join('SynthData', 'Set6', 'images',   'test')
    outlbl = os.path.join('SynthData', 'Set6', 'labels_ss','test')
    isfld =  os.path.join('SynthData', 'Set6', 'labels',   'test')


    os.makedirs(outfld, exist_ok=True)
    os.makedirs(outlbl, exist_ok=True)
    os.makedirs(isfld, exist_ok=True)

    thrds = []
    for i in range(threads):
        thrds.append(gen_parallel(fn_lists[i], outfld, outlbl, isfld, i==0))

    for t in thrds:
        t.start()

    for t in thrds:
        t.join()












