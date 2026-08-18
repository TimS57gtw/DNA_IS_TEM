import glob
import os

import cv2
import numpy as np

SYNTH_DIR = os.path.join('SynthData', 'MarkerSet2', 'images', 'train')
SYNTH_LBL_DIR = os.path.join('SynthData', 'MarkerSet2', 'labels', 'train')
REAL_CUTS_DIR = r'F:\Data\DimerAnalysis\Result\cuts'
OUT_DIR = os.path.join('SynthData', 'MarkerSet2_preview')


def draw_polygons(img_path, lbl_path, out_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(lbl_path) as f:
        for line in f.read().splitlines():
            parts = line.split(' ')
            coords = list(map(float, parts[1:]))
            xs = np.array(coords[0::2]) * w
            ys = np.array(coords[1::2]) * h
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 1)
    cv2.imwrite(out_path, img)


def collect_real_crops(n=8):
    paths = []
    for img_dir in sorted(glob.glob(os.path.join(REAL_CUTS_DIR, '*'))):
        if not os.path.isdir(img_dir):
            continue
        for crop in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
            if crop.endswith('_hist.png'):
                continue
            paths.append(crop)
            if len(paths) >= n:
                return paths
    return paths


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    synth_imgs = sorted(glob.glob(os.path.join(SYNTH_DIR, '*.png')))[:8]
    written = 0
    for img_path in synth_imgs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(SYNTH_LBL_DIR, stem + '.txt')
        if not os.path.isfile(lbl_path):
            continue
        draw_polygons(img_path, lbl_path, os.path.join(OUT_DIR, f'synth_{stem}.png'))
        written += 1
    assert written > 0, 'no synthetic samples found — run GenMarkerTrainingData.py first'

    real_written = 0
    for crop_path in collect_real_crops(n=8):
        out_path = os.path.join(OUT_DIR, f'real_{os.path.basename(os.path.dirname(crop_path))}_'
                                          f'{os.path.basename(crop_path)}')
        img = cv2.imread(crop_path)
        cv2.imwrite(out_path, img)
        real_written += 1

    print(f'self-check OK: wrote {written} synthetic + {real_written} real preview images -> {OUT_DIR}')
