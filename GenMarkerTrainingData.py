import os
import random
from multiprocessing import Process

import cv2
import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise
from tqdm import tqdm

import IS_TrainData as base

RESOLUTION = base.RESOLUTION  # 400, single-molecule tile size
MIN_MARKER_AREA = 9  # reject specks with contour area smaller than this after thresholding
SPOT_THRESHOLD = 2  # matches IS_TrainData.generate_image's spt_th


def _spot_mask_and_contribution(posx, posy, sigma_X, sigma_Y, rng_x, rng_y,
                                 height, theta, hfkt, shape):
    """Render one spot's Gaussian/cosine blob in isolation, returning both
    its contribution to the shared noise image and its own thresholded
    instance mask (same math as IS_TrainData.generate_image's spot loop,
    split so each spot keeps a separate identity instead of being summed
    into one shared mask before thresholding)."""
    a = np.cos(theta) ** 2 / (2 * sigma_X ** 2) + np.sin(theta) ** 2 / (2 * sigma_Y ** 2)
    b = np.sin(2 * theta) / (4 * sigma_X ** 2) - np.sin(2 * theta) / (4 * sigma_Y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_X ** 2) + np.cos(theta) ** 2 / (2 * sigma_Y ** 2)

    contribution = np.zeros(shape)
    for ii in range(int(posx) - rng_x, int(posx) + rng_x):
        for jj in range(int(posy) - rng_y, int(posy) + rng_y):
            h = height * hfkt(a * (ii - posx) ** 2 + 2 * b * (ii - posx) * (jj - posy)
                               + c * (jj - posy) ** 2)
            if 0 <= ii < shape[0] and 0 <= jj < shape[1]:
                contribution[ii, jj] += h

    spal = np.abs(contribution)
    kernel3 = np.ones((7, 7))
    spal = cv2.filter2D(spal, -1, kernel3)
    mask = (spal > SPOT_THRESHOLD).astype(np.uint8)
    return contribution, mask


def _mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_MARKER_AREA:
        return None
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)  # (n, 2) in (col, row) = (x, y) pixel coords


def generate_marker_sample(save_fn, label_fn, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    arr = np.zeros((RESOLUTION, RESOLUTION))

    octaves = np.random.randint(12, 28)
    noise_mu = np.abs(np.random.normal(0.5, 0.104))
    noise_mu = np.clip(noise_mu, 0.05, 0.75)
    noise_sig = 0.05 + 0.1 * np.random.random()
    sig = noise_sig * noise_mu
    sig = min(0.5, sig)

    coarse_res = 40
    perlin_x = PerlinNoise(octaves=octaves)
    coarse = np.array([[perlin_x([i / coarse_res, j / coarse_res]) for j in range(coarse_res)]
                        for i in range(coarse_res)])
    noise = cv2.resize(coarse, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_CUBIC)
    noise -= np.average(noise)
    noise /= np.std(noise)
    noise *= sig
    noise += noise_mu

    mol_height = np.random.uniform(0.1, max(0.2, 0.9 - noise_mu))
    mol_arr, label, _label_text = base.gen_DNA_shape()
    mol_arr = mol_arr * mol_height

    white_noise = np.random.normal(0.1, 0.05, arr.shape)

    no_spots = np.random.randint(0, 10)
    diam_x = np.random.normal(100 * RESOLUTION / 400, 20 * RESOLUTION / 400, no_spots)
    diam_y = np.random.normal(100 * RESOLUTION / 400, 20 * RESOLUTION / 400, no_spots)
    heights = [-np.random.uniform((mol_height + noise_mu) / 2, mol_height + noise_mu)
               for _ in range(no_spots)]

    # Restrict marker spots to the molecule's footprint plus a small margin
    # just outside it ("barely outside"), instead of scattering them over
    # the whole tile — dilate the molecule mask and sample positions from
    # the dilated area.
    margin_px = int((20 * RESOLUTION / 400) / 2) * 2 + 1
    dil_kernel = np.ones((margin_px, margin_px), np.uint8)
    mol_area = cv2.dilate(label.astype(np.uint8), dil_kernel)
    area_px = np.argwhere(mol_area > 0)

    spot_arr = np.zeros_like(arr)
    polygons = []

    for i in range(no_spots):
        if len(area_px) > 0:
            pos = area_px[np.random.randint(0, len(area_px))]
            posx, posy = pos[0], pos[1]
        else:
            posx = np.random.randint(0, RESOLUTION)
            posy = np.random.randint(0, RESOLUTION)

        sigma_X = np.sqrt(diam_x[i])
        sigma_Y = np.sqrt(diam_y[i])
        rng_x = int(3 * sigma_X)
        rng_y = int(3 * sigma_Y)
        theta = np.random.random() * 2 * np.pi
        if random.random() > 0.5:
            hfkt = lambda x: np.exp(-x)
        else:
            fk = np.random.uniform(0.5, 3)
            hfkt = lambda x, fk=fk: np.cos(fk * x) * np.exp(-x)

        contribution, mask = _spot_mask_and_contribution(
            posx, posy, sigma_X, sigma_Y, rng_x, rng_y, heights[i], theta, hfkt, arr.shape)
        spot_arr += contribution

        poly = _mask_to_polygon(mask)
        if poly is not None:
            polygons.append(poly)

    arr = noise + mol_arr + white_noise + spot_arr
    arr = np.clip(arr, 0, 1)

    os.makedirs(os.path.dirname(save_fn) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(label_fn) or '.', exist_ok=True)

    import matplotlib.pyplot as plt
    plt.imsave(save_fn, arr, vmin=0, vmax=1, cmap='gray')

    h, w = arr.shape
    with open(label_fn, 'w') as f:
        for poly in polygons:
            xs = poly[:, 0] / w
            ys = poly[:, 1] / h
            coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in zip(xs, ys))
            f.write(f'0 {coords}\n')

    return len(polygons)


class _gen_parallel(Process):
    """Generates a chunk of samples (a list of (index, seed) pairs) for one
    split in a subprocess, mirroring IS_TrainData.gen_parallel's pattern."""

    def __init__(self, indices, img_dir, lbl_dir, seed0):
        super().__init__()
        self.indices = indices
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.seed0 = seed0

    def run(self) -> None:
        for i in self.indices:
            fn = f'{str(i).zfill(6)}'
            generate_marker_sample(
                os.path.join(self.img_dir, fn + '.png'),
                os.path.join(self.lbl_dir, fn + '.txt'),
                seed=self.seed0 + i)


def generate_dataset(root, n_train=800, n_val=200, seed_base=1000):
    n_workers = max(1, os.cpu_count() or 8)

    for split, n, seed0 in (('train', n_train, seed_base), ('val', n_val, seed_base + n_train)):
        img_dir = os.path.join(root, 'images', split)
        lbl_dir = os.path.join(root, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        chunks = [[] for _ in range(n_workers)]
        for i in range(n):
            chunks[i % n_workers].append(i)

        procs = [_gen_parallel(chunk, img_dir, lbl_dir, seed0)
                 for chunk in tqdm(chunks) if chunk]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        failed = [p for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                f'{len(failed)} of {len(procs)} marker-generation worker(s) crashed '
                f'(exitcode != 0) while generating split {split!r} — '
                f'dataset under {root!r} is incomplete')

    # Recompute per-sample marker counts from the written label files, since
    # counts can no longer be collected synchronously from worker processes.
    counts = []
    for split, n in (('train', n_train), ('val', n_val)):
        lbl_dir = os.path.join(root, 'labels', split)
        for i in range(n):
            fn = os.path.join(lbl_dir, f'{str(i).zfill(6)}.txt')
            with open(fn) as f:
                counts.append(len(f.read().splitlines()))
    return counts


if __name__ == '__main__':
    import tempfile
    tmp = tempfile.mkdtemp()
    img_fn = os.path.join(tmp, 'sample.png')
    lbl_fn = os.path.join(tmp, 'sample.txt')

    n = generate_marker_sample(img_fn, lbl_fn, seed=0)
    assert os.path.isfile(img_fn), 'image not written'
    assert os.path.isfile(lbl_fn), 'label not written'
    with open(lbl_fn) as f:
        lines = f.read().splitlines()
    assert len(lines) == n, f'label line count {len(lines)} != returned count {n}'
    for line in lines:
        parts = line.split(' ')
        assert parts[0] == '0', f'unexpected class id {parts[0]}'
        coords = list(map(float, parts[1:]))
        assert len(coords) >= 6 and len(coords) % 2 == 0, f'bad polygon length {len(coords)}'
        assert all(0.0 <= c <= 1.0 for c in coords), 'coords not normalized to [0,1]'
    print(f'self-check OK: seed=0 produced {n} marker instances -> {img_fn}, {lbl_fn}')

    dataset_root = os.path.join('SynthData', 'MarkerSet2')
    counts = generate_dataset(dataset_root, n_train=4000, n_val=800)
    print(f'generated {len(counts)} samples -> {dataset_root} '
          f'(avg {np.mean(counts):.1f} markers/image, {sum(1 for c in counts if c == 0)} empty)')
