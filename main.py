import copy

from PIL import Image
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

def trafo_fn(inname):
    ext = inname[-4:]
    base = inname[:-4]
    forbidden_chars = [' ', '.', ',', ';', ':', '`', '/', '°', '(', ')']
    resname = ''
    for c in base:
        resname += c if c not in forbidden_chars else '_'
    resname += '.png'

    return resname

def trafo_fld(inname):
    forbidden_chars = [' ', '.', ',', ';', ':', '`', '/', '°', '(', ')']
    resname = ''
    for c in inname:
        resname += c if c not in forbidden_chars else '_'

    return resname

def hoshen_koppelmann(arr, minsize=None):
    """
    Implemenierung des Hoshen Koppelmann-Algorithmus
    """

    # Neues Array mit Padding links und oben
    padded_array = np.pad(arr, (1, 0))
    length_0 = arr.shape[0]
    length_1 = arr.shape[1]

    size = length_0 * length_1

    # Erstellung der Matrix fuer labels
    label = np.zeros(np.shape(padded_array), dtype=np.int32)

    # Zaehl-Array. None um Array-Indizes bei 1 zu beginnen
    n = [None]

    # Cluster-Index
    c = 1

    # Setze Label fuer linken und oberen Rand
    for i in range(length_1 + 1):
        label[0, i] = size
    for i in range(length_0+1):
        label[i, 0] = size

    # Fuktion zum Finden des "guten" labels
    def good_label(i):

        m = n[i]
        if m < 0:
            r = -m
            assert r >= 0
            m = n[r]

            while m < 0:
                r = -m
                assert r >= 0
                m = n[r]

            n[i] = -r

        else:
            r = i

        return r

    # Iteration über das Array und Fallunterscheidung je nach Besetzung der Plätze
    for i in tqdm(range(1, length_0 + 1), disable=True):
        for j in range(1, length_1 + 1):

            # Pixel ist leer
            if padded_array[i, j] == 0:
                label[i, j] = size
                continue

            # Oben und Unten sind nicht besetzt -> Neues Cluster
            if padded_array[i - 1, j] == 0 and padded_array[i, j - 1] == 0:
                label[i, j] = c
                n.append(1)
                c += 1
                continue

            # Nur Links ist besetzt
            if padded_array[i, j - 1] == 1 and padded_array[i - 1, j] == 0:
                l = good_label(label[i, j - 1])
                label[i, j] = l
                n[l] += 1


            # Nur oben ist besetzt
            elif padded_array[i, j - 1] == 0 and padded_array[i - 1, j] == 1:
                l = good_label(label[i - 1, j])
                n[l] += 1
                label[i, j] = l

            # Beide sind besetzt -> Kombiniere Cluster
            else:
                l = good_label(label[i, j - 1])
                u = good_label(label[i - 1, j])

                if u == l:
                    n[l] += 1
                    label[i, j] = l

                else:
                    if l < u:
                        n[l] = n[l] + n[u] + 1
                        n[u] = -l
                        label[i, j] = l
                    else:
                        assert l != u
                        n[u] = n[u] + n[l] + 1
                        n[l] = -u
                        label[i, j] = u

    # Umnummerierung
    cluster_sizes = [None]
    known_good_labels = [None]

    # Echte Label fuer jeden Pixel
    for i in tqdm(range(np.shape(label)[0]), disable=True):
        for j in range(np.shape(label)[1]):
            if padded_array[i, j] == 1:
                gl = good_label(label[i, j])
                if gl in known_good_labels:
                    ci = known_good_labels.index(gl)
                    label[i, j] = ci
                else:
                    known_good_labels.append(gl)
                    cluster_sizes.append(n[gl])
                    label[i, j] = len(known_good_labels) - 1
            else:
                label[i, j] = -1

    # Karte Mit Cluster-Groessen
    cs_map = np.zeros(np.shape(label))
    for i in range(np.shape(label)[0]):
        for j in range(np.shape(label)[1]):
            if padded_array[i, j] == 0:
                cs_map[i, j] = 0
            else:
                gl = label[i, j]
                cs_map[i, j] = cluster_sizes[gl]

    # Abschneiden der Array zum Eliminieren des Paddings
    labels = label[1:, 1:]

    if minsize is not None:
        # plt.imshow(label)
        # plt.title("labels")
        # # plt.show(block=True)
        # print("cluster_sizes: ", cluster_sizes)
        drop = []
        for i in range(1, len(cluster_sizes)):
            if cluster_sizes[i] < minsize:
                drop.append(i)
        for elem in drop:
            labels[labels == elem] = -1
        # plt.imshow(label)
        # plt.title("After Drop")
        # # plt.show(block=True)

        # Umnummerieren
        vals = np.unique(label)
        # print("Unique: ", vals)
        for new, elem in enumerate(vals):
            if elem < 0:
                continue
            labels[labels == elem] = new

        # print("New unique: ", np.unique(labels))
        # plt.imshow(labels)
        # plt.title("Afer renum")
        # # plt.show(block=True)

    return labels

total_idx = 1

def analyze_file(infile, outfile):
    global total_idx

    COMPLEX_FILL = True

    arr = np.array(Image.open(infile)).astype(float)

    # print(arr)
    # plt.imshow(arr)
    # plt.show()

    ss = 1
    # cut logo
    arr = arr[:2048, :]

    # Subsample
    arr = arr[::ss, ::ss]

    # Find high threshold
    vals = arr.flatten()
    # plt.hist(vals, bins=100)
    # plt.show()


    # replace dimer
    th_spots = 175
    medi = np.median(arr)
    arr_noSpot = copy.deepcopy(arr)
    arr_Spot = copy.deepcopy(arr)

    arr_noSpot[arr_noSpot > th_spots] = medi
    arr_Spot[arr_Spot < th_spots] = 0


    # plt.imshow(arr_noSpot)
    # plt.title('No Spots')
    # plt.show()


    # high-convolve to find molecules
    ks = 101
    mid = 51
    sig = 40 / ss
    kernel = np.zeros((ks, ks))
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x, y] = np.exp(- np.sqrt((x-mid)**2 + (y - mid)**2)/ sig )

    kernel /= np.sum(kernel)
    # plt.imshow(kernel)
    # plt.title('Large Kernel')
    # plt.show()

    arr_spots = cv2.filter2D(arr_Spot, -1, kernel)

    # plt.imshow(arr_spots)
    # plt.title('Spots')
    # plt.show()

    thHK = 5
    binary = np.zeros_like(arr_spots)
    binary[arr_spots > thHK] = 1

    # plt.imshow(binary)
    # plt.title('binary')
    # plt.show()

    #print('Start HK1')
    clusters = hoshen_koppelmann(binary)
    # print('End HK1')


    border_vals = []
    for i in range(clusters.shape[0]):
        for j in [0, -1]:
            if clusters[i, j] > 0 and clusters[i, j] not in border_vals:
                border_vals.append(clusters[i, j])

    for i in [0, -1]:
        for j in range(clusters.shape[1]):
            if clusters[i, j] > 0 and clusters[i, j] not in border_vals:
                border_vals.append(clusters[i, j])

    for bv in border_vals:
        clusters[clusters == bv] = -1


    vals = np.unique(clusters)
    vals = [x for x in vals if x > 0]
    sizes = [len(np.argwhere(clusters == x)) for x in vals]


    smallvals = []
    normsize = []
    for v, s in zip(vals, sizes):
        if s < 100:
            smallvals.append(v)
        else:
            normsize.append(s)

    for sv in smallvals:
        clusters[clusters == sv] = -1

    # plt.imshow(clusters)
    # plt.title("Filter Small")
    # plt.show()

    medsize = np.median(normsize)

    lowth = 0.5
    hith = 2

    # print('Ths: ', lowth * medsize , hith * medsize)

    for v, s in zip(vals, sizes):
        if not lowth * medsize < s < hith * medsize:
            smallvals.append(v)
        else:
            normsize.append(s)


    for sv in smallvals:
        clusters[clusters == sv] = -1

    # plt.imshow(clusters)
    # plt.title("Filter Sizes")
    # plt.show()


    vals = np.unique(clusters)
    vals = [x for x in vals if x > 0]

    centers = []

    for val in vals:
        x = []
        y = []
        pts = np.argwhere(clusters == val)
        # print(pts)
        for pt in pts:
            x.append(pt[1])
            y.append(pt[0])

        centers.append(np.array([np.median(x), np.median(y)]))

    # print(centers)


    for cent in centers:
        xl = int(np.floor(cent[0] - (200 / ss)))
        xr = int(np.ceil(cent[0] + (200 / ss)))
        yl = int(np.floor(cent[1] - (200 / ss)))
        yr = int(np.ceil(cent[1] + (200 / ss)))

        locarr = arr[max(0, yl):min(arr.shape[0]-1, yr), max(0, xl):min(arr.shape[1]-1, xr)]
        locarr_remSPT = copy.deepcopy(locarr)

        # plt.imshow(locarr)
        # plt.title('Loc')
        # plt.show()

        locarr_ns = np.zeros_like(locarr)
        locarr_ns[locarr > th_spots] = 1

        # plt.imshow(locarr_ns)
        # plt.show()

        cls = hoshen_koppelmann(locarr_ns, minsize=int(40 / ss))

        # plt.imshow(cls)
        # plt.show()


        # Filter each spot by surrounding values

        vals = [x for x in np.unique(cls) if x > 0]
        for val in vals:

            if COMPLEX_FILL:

                mbenl = np.zeros_like(cls, dtype=int)

                pos = np.argwhere(cls == val)
                bm = np.zeros_like(cls, dtype=int)
                bm[cls == val] = 1

                max0 = mbenl.shape[0] - 1
                max1 = mbenl.shape[1] - 1

                for posi in pos:
                    for i in [-int(8/ss), 0, int(8/ss)]:
                        for j in [-int(8/ss), 0, int(8/ss)]:
                            mbenl[min(max(0, posi[0] + i), max0), min(max(0, posi[1] + j), max1)] = 1


                # plt.imshow(bm)
                # plt.title('bm')
                # plt.show()
#
                # plt.imshow(mbenl)
                # plt.title("englarged")
                # plt.show()



                diff = mbenl - bm
                # plt.imshow(diff)
                # plt.title('diff')
                # plt.show()
                nz_pos = np.argwhere(diff == 1)
                vals = []
                for p in nz_pos:
                    vals.append(locarr[p[0], p[1]])

                fill = np.average(vals)

            else:
                fill = medi

            locarr_remSPT[cls == val] = fill


        # plt.imshow(locarr_remSPT)
        # plt.title("Locarr RemSpot")
        # plt.show()


        locarr_rs_img = locarr_remSPT.astype(np.uint8)

        ks = 21
        mid = 11
        sig = 10 / ss
        kernel10 = np.zeros((ks, ks))
        for x in range(kernel10.shape[0]):
            for y in range(kernel10.shape[1]):
                kernel10[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / sig)

        sig = 5 / ss
        kernel5 = np.zeros((ks, ks))
        for x in range(kernel5.shape[0]):
            for y in range(kernel5.shape[1]):
                kernel5[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / sig)

        sig = 2 / ss
        kernel2 = np.zeros((ks, ks))
        for x in range(kernel2.shape[0]):
            for y in range(kernel2.shape[1]):
                kernel2[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / sig)

        sig = 1 / ss
        kernel1 = np.zeros((ks, ks))
        for x in range(kernel1.shape[0]):
            for y in range(kernel1.shape[1]):
                kernel1[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / sig)



        kernel10 /= np.sum(kernel10)
        kernel5 /= np.sum(kernel5)
        kernel2 /= np.sum(kernel2)
        kernel1 /= np.sum(kernel1)

        # fig, axs = plt.subplots(1, 4)
        # axs[0].imshow(kernel10)
        # axs[0].set_title(10)
        # axs[1].imshow(kernel5)
        # axs[1].set_title(5)
        # axs[2].imshow(kernel2)
        # axs[2].set_title(2)
        # axs[3].imshow(kernel1)
        # axs[3].set_title(1)
        # plt.show()

        # locarr_cnv10 = cv2.filter2D(locarr_rs_img, -1, kernel10)
        locarr_cnv5 = cv2.filter2D(locarr_rs_img, -1, kernel5)
        # locarr_cnv2 = cv2.filter2D(locarr_rs_img, -1, kernel2)
        # locarr_cnv1 = cv2.filter2D(locarr_rs_img, -1, kernel1)


        # plt.imshow(locarr_cnv10)
        # plt.title('Cnv10')
        # plt.show()
        # plt.imshow(locarr_cnv5)
        # plt.title('Cnv5')
        # plt.show()
        # plt.imshow(locarr_cnv2)
        # plt.title('Cnv2')
        # plt.show()
        # plt.imshow(locarr_cnv1)
        # plt.title('Cnv1')
        # plt.show()
#
        cnv_arr = locarr_cnv5.astype(float)


        cnv_arr *= -1

        cnv_arr -= np.amin(cnv_arr)
        cnv_arr /= np.amax(cnv_arr)

        # plt.imshow(cnv_arr)
        # plt.show()

        plt.imsave(os.path.join(r'D:\seifert\PycharmProjects\LiuDimer\RealCrops', str(total_idx).zfill(6) + '.png'), cnv_arr, cmap='gray')
        total_idx += 1




def analyze_fodler(infld, outfld, totalfld=None):
    os.makedirs(outfld, exist_ok=True)
    infiles = [os.path.join(infld, x) for x in os.listdir(infld) if x.split('.')[-1] == 'tif']
    outfiles = [os.path.join(outfld, trafo_fn(os.path.basename(x))) for x in infiles]
    further_folders = [x for x in os.listdir(infld) if os.path.isdir(os.path.join(infld, x))]
    while len(further_folders) > 0:
        ff = further_folders.pop()
        fffs = [os.path.join(infld, ff, x) for x in os.listdir(os.path.join(infld, ff))]
        for f in fffs:
            if os.path.isdir(f):
                further_folders.append(f)
            elif f.split('.')[-1] == 'tif':
                os.makedirs(os.path.join(outfld, trafo_fld(ff)), exist_ok=True)
                infiles.append(os.path.join(infld, ff, os.path.basename(f)))
                outfiles.append(os.path.join(outfld, trafo_fld(ff), trafo_fn(os.path.basename(f))))

    for inf, outf in tqdm(zip(infiles, outfiles), desc='Modify files', total=len(infiles)):

        if os.path.isfile(outf):
            print("Already exists: ", outf)
            continue

        try:
            analyze_file(inf, outf)
        except Exception as e:
            raise e
            print(e)

    if totalf is not None:
        os.makedirs(totalf, exist_ok=True)
        with open(os.path.join(totalf, '00_filenames.csv'), 'w') as f:
            for i, of in enumerate(outfiles):
                shutil.copy(of, os.path.join(totalf, f'{i}.png'))

                f.write(f"{of};{os.path.join(totalf, f'{i}.png')}\n")

if __name__ == '__main__':
    infld = r'D:\seifert\PycharmProjects\LiuDimer\Data'
    outfld = r'D:\seifert\PycharmProjects\LiuDimer\Output\Test2'
    totalfld = r'D:\seifert\PycharmProjects\LiuDimer\Output\all'

    analyze_fodler(infld, outfld, totalfld=None)




