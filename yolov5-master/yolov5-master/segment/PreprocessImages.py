import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from multiprocessing import Process, Pool
def pp(pair):
    fin, fout = pair
    if os.path.isfile(fout):
        return
    arr = np.array(Image.open(fin)).astype(float)
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]

    pct = np.percentile(arr, 5)
    maxp = np.percentile(arr, 95)

    # vals = arr.flatten()
    # plt.hist(vals)
    # plt.show()
    # plt.imshow(arr)
    # plt.title(f'pct: {pct}')
    # plt.show()
    arr = np.clip(arr, pct, maxp)
    arr -= np.amin(arr)
    arr /= np.amax(arr)    # plt.imshow(arr)
    # plt.title(f'pct: {pct}')
    # plt.show()
    xs = np.average(arr, axis=0)
    ys = np.average(arr, axis=1)
    valsX = [x for x in range(len(xs))]
    valsY = [x for x in range(len(ys))]
    lrX = linregress(valsX, xs)
    lrY = linregress(valsY, ys)
    fx = lambda x : lrX.intercept + lrX.slope * x
    fy = lambda x : lrY.intercept + lrY.slope * x
    fitX = [fx(x) for x in valsX]
    fitY = [fy(x) for x in valsY]
    # plt.scatter(valsX, xs, label='X')
    # plt.scatter(valsY, ys, label='Y')
    # plt.plot(valsX, fitX, label='fx')
    # plt.plot(valsY, fitY, label='fy')
    # plt.legend()
    # plt.show()
    mgx, mgy = np.meshgrid(valsX, valsY)
    f = lambda x, y: lrX.slope * x + lrY.slope * y
    f = np.vectorize(f)
    plane = f(mgx, mgy)
    # print(plane)
    # plt.imshow(plane)
    # plt.show()
    arr -= plane
    # fig, axs = plt.subplots(1,4)
    # axs[0].imshow(arr)
    # axs[1].imshow(plane)
    # axs[2].imshow(arrFL)
    # axs[3].imshow(arrFL - arr)
    # plt.show()
    plt.imsave(fout, arr, cmap='gray')

def pp_list(list, distq=True):
    for elem in tqdm(list, disable=distq):
        pp(*elem)


def pp_fld(fld_in, fld_out):
    thrds = 15
    infs = []
    oufns = []
    os.makedirs(fld_out)
    for elem in os.listdir(fld_in):
        assert os.path.isdir(os.path.join(fld_in, elem))
        os.makedirs(os.path.join(fld_out, elem))
        infs += [os.path.join(fld_in, elem, x) for x in os.listdir(os.path.join(fld_in, elem))]
        bns = [os.path.basename(x) for x in infs]
        oufns += [os.path.join(fld_out, elem, x) for x in bns]

    tasks = []
    for i in range(thrds):
        tasks.append([])

    idx = 0
    for fin, fout in tqdm(zip(infs, oufns), total=len(infs)):
        tasks[idx%thrds].append((fin, fout))
        idx += 1

    ts = []
    for t in range(thrds):
        p = Process(target=pp_list, args=(tasks[t], t!=0))
        ts.append(p)

    for p in ts:
        p.start()
    for p in ts:
        p.join()

def pp_list_parallel(lst):


    with Pool(20) as p:
        for _ in tqdm(p.imap_unordered(pp, lst), total=len(lst), desc='Parallel PP'):
            pass

    return

    thrds = 15
    tasks = []
    for i in range(thrds):
        tasks.append([])




    idx = 0
    for fin, fout in tqdm(lst, total=len(lst), desc='Parallel PP'):
        tasks[idx%thrds].append((fin, fout))
        idx += 1

    ts = []
    for t in range(thrds):
        p = Process(target=pp_list, args=(tasks[t], t!=0))
        ts.append(p)

    for p in ts:
        p.start()
    for p in ts:
        p.join()





if __name__ == '__main__':
    fld_in =  r'D:\seifert\PycharmProjects\LiuDimer\SynthData\Set5IS_400\images'
    fld_out = r'C:\Users\seifert\Pictures\Temp\images_pp'
    pp_fld(fld_in, fld_out)