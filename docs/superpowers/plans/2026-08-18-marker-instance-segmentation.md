# Marker Instance Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learned instance segmentation for gold markers inside DNA-molecule
crops, replacing the hand-tuned LoG peak detector in `find_markers.py`.

**Architecture:** Fork the existing synthetic molecule-image generator
(`IS_TrainData.py`) into a marker-labeling variant that keeps each spot's
own blob mask as a separate YOLO-seg polygon instance instead of merging
them into one semantic mask. Train a YOLOv5-seg model on that data (user
runs training, out of scope for these tasks). Add an inference function to
`find_markers.py` that swaps in for the LoG detector when trained weights
are given.

**Tech Stack:** Python, numpy, opencv-python (`cv2`), Pillow, matplotlib,
shapely, perlin-noise (all already used by `IS_TrainData.py`), PyTorch +
the vendored `yolov5-master` repo for training/inference.

**Spec:** `docs/superpowers/specs/2026-08-18-marker-instance-segmentation-design.md`

## Global Constraints

- Marker label format is YOLO-seg: `<class> x1 y1 x2 y2 ... xn yn`, one line
  per instance, coordinates normalized to [0, 1] by image width/height,
  class `0` = `Marker` (single class).
- Generated image pixels must stay byte-identical to `IS_TrainData.py`'s
  `generate_image()` output — only label extraction is new, the rendered
  image is unchanged.
- No pytest infra exists in this repo; validate with `assert`-based
  self-checks run via each script's `if __name__ == '__main__':` block, per
  repo convention (`IS_TrainData.py`, `GenTrainingData.py`).
- `NM_P_PX = 0.8431` (pixel-to-nm scale) already defined in `find_markers.py`
  — do not redefine it.

---

### Task 1: Marker polygon generator (`GenMarkerTrainingData.py`)

**Files:**
- Create: `GenMarkerTrainingData.py`

**Interfaces:**
- Produces: `generate_marker_sample(save_fn: str, label_fn: str, seed: int | None = None) -> int`
  — writes the PNG to `save_fn` and the YOLO-seg label txt to `label_fn`
  (empty file if zero markers survive), returns the number of marker
  instances written. Used by Task 2's batch driver and by Task 1's own
  self-check.

- [ ] **Step 1: Write `GenMarkerTrainingData.py` with the forked generator**

```python
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
```

- [ ] **Step 2: Run the self-check**

Run: `python GenMarkerTrainingData.py`
Expected: prints `self-check OK: seed=0 produced N marker instances -> ...` and
exits 0 (no `AssertionError`). `N` depends on the seeded RNG draw — any
non-negative count that passes the asserts is fine.

- [ ] **Step 3: Commit**

```bash
git add GenMarkerTrainingData.py
git commit -m "feat: add synthetic marker instance-segmentation data generator"
```

---

### Task 2: Batch generation, dataset yaml, and visual sanity check

**Files:**
- Modify: `GenMarkerTrainingData.py` (add batch driver + `__main__` wiring)
- Create: `yolov5-master/yolov5-master/data/SetMarkerIS.yaml`
- Create: `visualize_marker_training_data.py`

**Interfaces:**
- Consumes: `generate_marker_sample(save_fn, label_fn, seed=None) -> int`
  from Task 1.
- Produces: on-disk dataset under `SynthData/MarkerSet1/{images,labels}/{train,val}`
  and `SynthData/MarkerSet1_preview/*.png` (visualization output), consumed
  by the user's own `train.py` run (not by later tasks).

- [ ] **Step 1: Add a batch driver to `GenMarkerTrainingData.py`**

Append below the `if __name__ == '__main__':` self-check block from Task 1
(keep the self-check as the first thing that runs, then generate the batch):

```python
def generate_dataset(root, n_train=800, n_val=200, seed_base=1000):
    counts = []
    for split, n, seed0 in (('train', n_train, seed_base), ('val', n_val, seed_base + n_train)):
        img_dir = os.path.join(root, 'images', split)
        lbl_dir = os.path.join(root, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            fn = f'{str(i).zfill(6)}'
            n_markers = generate_marker_sample(
                os.path.join(img_dir, fn + '.png'),
                os.path.join(lbl_dir, fn + '.txt'),
                seed=seed0 + i)
            counts.append(n_markers)
    return counts
```

- [ ] **Step 2: Wire dataset generation behind the self-check in `__main__`**

Replace the existing `if __name__ == '__main__':` block in
`GenMarkerTrainingData.py` with:

```python
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

    dataset_root = os.path.join('SynthData', 'MarkerSet1')
    counts = generate_dataset(dataset_root, n_train=800, n_val=200)
    print(f'generated {len(counts)} samples -> {dataset_root} '
          f'(avg {np.mean(counts):.1f} markers/image, {sum(1 for c in counts if c == 0)} empty)')
```

- [ ] **Step 3: Run it**

Run: `python GenMarkerTrainingData.py`
Expected: self-check line, then `generated 1000 samples -> SynthData/MarkerSet1
(avg X.X markers/image, Y empty)`. Takes a few minutes (1000 Perlin-noise
400x400 renders). Verify `SynthData/MarkerSet1/images/train` has 800 PNGs
and `SynthData/MarkerSet1/labels/train` has 800 txt files (some may be
empty — that's expected).

- [ ] **Step 4: Create the dataset yaml**

```yaml
# yolov5-master/yolov5-master/data/SetMarkerIS.yaml
path: SynthData/MarkerSet1
train: images/train
val: images/val

names:
  0: Marker
```

(Adjust `path` to an absolute path if training is run from a different
working directory than the repo root — mirrors `SetV5IS_400_PP.yaml`'s
`path:` convention.)

- [ ] **Step 5: Write the visualization script**

```python
# visualize_marker_training_data.py
import glob
import os

import cv2
import numpy as np

SYNTH_DIR = os.path.join('SynthData', 'MarkerSet1', 'images', 'train')
SYNTH_LBL_DIR = os.path.join('SynthData', 'MarkerSet1', 'labels', 'train')
REAL_CUTS_DIR = r'F:\Data\DimerAnalysis\Result\cuts'
OUT_DIR = os.path.join('SynthData', 'MarkerSet1_preview')


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
```

- [ ] **Step 6: Run it and inspect the images**

Run: `python visualize_marker_training_data.py`
Expected: `self-check OK: wrote 8 synthetic + N real preview images -> SynthData/MarkerSet1_preview`.
Open a few `synth_*.png` and `real_*.png` from that folder and compare —
polygon outlines in the synthetic ones should sit on dark round marker
blobs similar in size/shape to the dark spots in the real crops.

- [ ] **Step 7: Commit**

```bash
git add GenMarkerTrainingData.py visualize_marker_training_data.py \
  "yolov5-master/yolov5-master/data/SetMarkerIS.yaml"
git commit -m "feat: batch-generate marker dataset and visualize against real crops"
```

*(`SynthData/` output is generated data, not source — do not commit it;
add `SynthData/` to `.gitignore` if it isn't already ignored.)*

---

### Task 3: Model inference integration in `find_markers.py`

**Files:**
- Modify: `find_markers.py`

**Interfaces:**
- Consumes: nothing from Task 1/2 at runtime (only the dataset/yaml they
  produced, used offline for training — training itself is run by the user,
  not by this task).
- Produces: `detect_markers_yolo(crop_rgb, model, device, imgsz=400,
  conf_thres=0.25, iou_thres=0.45) -> list[tuple[float, float, float, float]]`
  (val, y, x, r) — same tuple shape `detect_in_crop` returns, so
  `find_markers_in_cuts` can call either interchangeably.

- [ ] **Step 1: Add the YOLO-seg detector function**

Add near the top of `find_markers.py`, after the existing imports (the
`DIMER_REPO` sys.path insert must stay above this since `models.common`
etc. come from the vendored `yolov5-master` repo, which needs its own
sys.path entry):

```python
YOLOV5_ROOT = r'G:\seife\PycharmG\DNA_IS_TEM\yolov5-master\yolov5-master'
if YOLOV5_ROOT not in sys.path:
    sys.path.insert(0, YOLOV5_ROOT)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device
import torch


def load_marker_model(weights, device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    return model, device


def detect_markers_yolo(crop_rgb, model, device, imgsz=400, conf_thres=0.25, iou_thres=0.45):
    """Run the trained marker YOLOv5-seg model on one molecule crop.
    Returns (val, y, x, r) tuples matching detect_in_crop's output shape,
    val = detection confidence, r = sqrt(area / pi) from the predicted mask."""
    h0, w0 = crop_rgb.shape[:2]
    img = cv2.resize(crop_rgb, (imgsz, imgsz))
    im = torch.from_numpy(img).to(device).float().permute(2, 0, 1) / 255
    im = im.unsqueeze(0)

    pred, proto = model(im, augment=False)[:2]
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000, nm=32)
    det = pred[0]
    if not len(det):
        return []

    masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], (h0, w0)).round()

    results = []
    for seg, conf in zip(masks2segments(masks), det[:, 4].tolist()):
        poly = seg.copy().astype(np.float32)
        poly[:, 0] *= w0 / imgsz
        poly[:, 1] *= h0 / imgsz
        area = cv2.contourArea(poly.astype(np.float32))
        if area <= 0:
            continue
        m = poly.mean(axis=0)
        r = float(np.sqrt(area / np.pi))
        results.append((float(conf), float(m[1]), float(m[0]), r))
    return results
```

- [ ] **Step 2: Wire it into `find_markers_in_cuts`**

Modify the signature and body (`find_markers.py`, currently
`def find_markers_in_cuts(result_dir, out_dir, margin=25):`):

```python
def find_markers_in_cuts(result_dir, out_dir, margin=25, weights=None):
    """Find markers inside each segmented cut. Uses the trained YOLOv5-seg
    marker model when `weights` is given; otherwise falls back to the
    multi-scale LoG peak detector (calibrate_sigma/detect_in_crop) below."""
    os.makedirs(out_dir, exist_ok=True)
    items = collect_processed_crops(result_dir, margin)
    if not items:
        print('No cuts found to search for markers.')
        return

    model = device = None
    sigmas = None
    if weights is not None:
        model, device = load_marker_model(weights)
        print(f'Using YOLOv5-seg marker model: {weights}')
    else:
        calibrate_sigma(items)
        sigmas = [2, 3, 4, 5, 7, 9]
        print(f'Using sigma set: {sigmas} (marker radius ~'
              f'{sigmas[0] * np.sqrt(2):.1f}-{sigmas[-1] * np.sqrt(2):.1f}px)')

    rows = []
    for it in items:
        if model is not None:
            crop_rgb = cv2.cvtColor((it['norm'] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            peaks = detect_markers_yolo(crop_rgb, model, device)
        else:
            peaks = detect_in_crop(it['inv'], it['region'], sigmas)

        im = Image.fromarray((it['norm'] * 255).astype(np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(im)
        for val, y, x, r in peaks:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
            rows.append([it['img_id'], it['idx'], it['conf'], it['x0'] + x, it['y0'] + y, r])

        img_out_dir = os.path.join(out_dir, it['img_id'])
        os.makedirs(img_out_dir, exist_ok=True)
        im.save(os.path.join(img_out_dir, f"{it['idx']}_markers.png"))

    with open(os.path.join(out_dir, 'markers.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_id', 'crop_id', 'seg_conf', 'x_px', 'y_px', 'r_px'])
        writer.writerows(rows)

    print(f'{len(rows)} markers found across {len(items)} cuts -> {out_dir}')
```

Note `detect_in_crop`'s tuples are already `(val, y, x, s)` where `s` is the
LoG sigma, not a pixel radius — check the existing draw loop
(`r = s * np.sqrt(2)`) before this change: `detect_markers_yolo` returns an
actual pixel radius `r` directly, so the unified loop above must not
re-apply `* np.sqrt(2)`. To keep both code paths producing values the draw
loop can use identically, change `detect_in_crop` to return `r = s *
np.sqrt(2)` directly instead of `s`:

```python
def detect_in_crop(inv, region, sigmas, min_dist=6, top_frac=0.6, max_n=2):
    candidates = []
    for s in sigmas:
        for val, y, x in log_peaks(inv, region, s, threshold_rel=0.2):
            candidates.append((val, y, x, s))
    peaks = nms_peaks(candidates, min_dist, top_frac, max_n)
    return [(val, y, x, s * np.sqrt(2)) for val, y, x, s in peaks]
```

- [ ] **Step 3: Update the `__main__` block to accept an optional weights path**

```python
if __name__ == '__main__':
    result_dir = r'F:\Data\DimerAnalysis\Result'
    out_dir = os.path.join(result_dir, 'cuts')
    analysis_dir = os.path.join(result_dir, 'analysis')
    marker_weights = None  # set to a trained best.pt path to use the YOLO detector
    make_cuts(result_dir, out_dir)
    find_markers_in_cuts(result_dir, out_dir, weights=marker_weights)
    run_statistical_analysis(result_dir, out_dir, analysis_dir)
```

- [ ] **Step 4: Verify the LoG fallback path still runs unchanged**

Run: `python find_markers.py`
Expected: same behavior as before this task (no `weights` set, so
`detect_in_crop`/LoG path runs) — completes without exceptions and prints
`N markers found across M cuts -> ...`. This confirms the refactor of
`detect_in_crop`'s return radius didn't break the existing default path.
The `detect_markers_yolo` path can't be exercised yet — no trained weights
exist until the user runs `segment/train.py` on the Task 2 dataset.

- [ ] **Step 5: Commit**

```bash
git add find_markers.py
git commit -m "feat: add YOLOv5-seg marker detector, wire in behind --weights"
```

---

## Self-Review Notes

- Spec coverage: data generator (Task 1), dataset yaml + visualization
  (Task 2), inference integration with LoG fallback (Task 3) — all spec
  sections have a task. Training itself intentionally has no task (spec
  scopes it to the user).
- Placeholder scan: no TBD/TODO; all code blocks are complete, runnable.
- Type consistency: `detect_in_crop` and `detect_markers_yolo` both return
  `(val, y, x, r)` 4-tuples with `r` a pixel radius (Task 3 Step 2 makes
  this explicit and fixes the pre-existing `s`-vs-`r` mismatch).
