import os
import random

import cv2
import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise

import IS_TrainData as base

RESOLUTION = base.RESOLUTION  # 400, single-molecule tile size
MIN_MARKER_PIXELS = 9  # reject specks smaller than this after thresholding
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
    if cv2.contourArea(contour) < MIN_MARKER_PIXELS:
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

    perlin_x = PerlinNoise(octaves=octaves)
    noise = [[perlin_x([i / RESOLUTION, j / RESOLUTION]) for j in range(RESOLUTION)]
              for i in range(RESOLUTION)]
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

    non_zero_indices = np.nonzero(label)
    average_position = np.mean(np.column_stack(non_zero_indices), axis=0)

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    borderT = cv2.filter2D(label.astype(np.uint8), -1, kernel)
    k2s = int((25 * RESOLUTION / 400) / 2) * 2 + 1
    kernel2 = np.ones((k2s, k2s))
    border = cv2.filter2D(borderT.astype(np.uint8), -1, kernel2)
    brd = np.argwhere(border > 0)

    spot_arr = np.zeros_like(arr)
    polygons = []

    for i in range(no_spots):
        ordered = np.random.random() < 0.7
        on_border = np.random.random() < 0.5
        if ordered:
            if on_border and len(brd) > 0:
                pos = brd[np.random.randint(0, len(brd))]
                posx, posy = pos[0], pos[1]
            else:
                posy = np.random.normal(average_position[1], RESOLUTION / 5)
                posx = np.random.normal(average_position[0], RESOLUTION / 5)
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
