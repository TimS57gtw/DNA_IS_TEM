import csv
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter, gaussian_laplace, binary_dilation
from skimage.feature import peak_local_max

DIMER_REPO = r'G:\seife\PycharmG\automatic-dimer-analysis-tim'
if DIMER_REPO not in sys.path:
    sys.path.insert(0, DIMER_REPO)

from classes.find_closest_particles import find_closest_particles
from classes.find_monomers import find_monomer
from classes.find_dimers import find_dimer, check_for_equidistant_dimers
from classes.find_agglomerates import find_agglomerates
from classes.find_ith_smallest_distance import find_ith_smallest_distance

YOLOV5_ROOT = r'G:\seife\PycharmG\DNA_IS_TEM\yolov5-master\yolov5-master'
if YOLOV5_ROOT not in sys.path:
    sys.path.insert(0, YOLOV5_ROOT)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device
import torch

NM_P_PX = 0.8431  # from yolov5-master/segment/predictTEM.py


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


def read_polyRESC(path):
    with open(path) as f:
        parts = f.read().split()
    coords = list(map(float, parts[1:-1]))
    xs = coords[0::2]
    ys = coords[1::2]
    conf = float(parts[-1])
    return xs, ys, conf


def bbox(w, h, xs, ys, margin):
    x0 = max(0, int(min(xs)) - margin)
    x1 = min(w, int(max(xs)) + margin)
    y0 = max(0, int(min(ys)) - margin)
    y1 = min(h, int(max(ys)) + margin)
    return x0, y0, x1, y1


def poly_mask(shape, xs, ys):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(list(zip(xs, ys)), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def save_inout_histogram(gray_crop, mask_crop, out_path, idx):
    inside = gray_crop[mask_crop == 1]
    outside = gray_crop[mask_crop == 0]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(outside, bins=50, alpha=0.6, density=True, label=f'outside (n={outside.size})')
    ax.hist(inside, bins=50, alpha=0.6, density=True, label=f'inside (n={inside.size})')
    ax.set_title(f'Instance {idx}')
    ax.set_xlabel('pixel value')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def denoise_for_log(gray_crop, mask_crop, median_size=3, sigma=1.0, dilate_iters=5):
    """Light speckle removal + per-crop contrast normalization against the
    segmented region's own intensity range, dark markers inverted to bright
    so LoG maxima = marker centers. Everything outside the (dilated) region
    is zeroed so responses only come from inside the segmented blob."""
    sm = median_filter(gray_crop.astype(np.float32), size=median_size)
    sm = gaussian_filter(sm, sigma=sigma)
    region = binary_dilation(mask_crop.astype(bool), iterations=dilate_iters)
    lo, hi = np.percentile(sm[region], [2, 98])
    norm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
    inv = 1 - norm
    inv[~region] = 0
    return norm, inv, region


def in_region(region, y, x):
    yi, xi = int(round(y)), int(round(x))
    return 0 <= yi < region.shape[0] and 0 <= xi < region.shape[1] and region[yi, xi]


def collect_processed_crops(result_dir, margin):
    crops_root = os.path.join(result_dir, 'Crops')
    input_dir = os.path.join(result_dir, 'input')
    items = []
    for img_id in sorted(os.listdir(crops_root)):
        input_path = os.path.join(input_dir, img_id + '.png')
        if not os.path.isfile(input_path):
            continue
        input_img = Image.open(input_path).convert('L')
        gray = np.array(input_img)
        inst_dir = os.path.join(crops_root, img_id)
        for idx in sorted(os.listdir(inst_dir), key=int):
            polyfiles = glob.glob(os.path.join(inst_dir, idx, 'polyRESC*.txt'))
            if not polyfiles:
                continue
            xs, ys, conf = read_polyRESC(polyfiles[0])
            x0, y0, x1, y1 = bbox(*input_img.size, xs, ys, margin)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            mask = poly_mask(gray.shape, xs, ys)[y0:y1, x0:x1]
            norm, inv, region = denoise_for_log(gray[y0:y1, x0:x1], mask)
            items.append({'img_id': img_id, 'idx': idx, 'norm': norm, 'inv': inv, 'region': region,
                          'x0': x0, 'y0': y0, 'conf': conf})
    return items


def log_peaks(inv, region, sigma, threshold_rel=0.2):
    resp = -gaussian_laplace(inv, sigma) * (sigma ** 2)
    resp[~region] = 0
    peaks = peak_local_max(resp, min_distance=max(1, int(sigma)), threshold_rel=threshold_rel,
                            exclude_border=False)
    return [(resp[y, x], y, x) for y, x in peaks if region[y, x]]


def calibrate_sigma(items, sigma_search=(2, 3, 4, 5, 7, 9, 12)):
    """Scan every cut across a broad scale range and report the spread of
    each crop's strongest in-region response, purely as a sanity check.
    NOTE: the single strongest response per crop is consistently biased
    toward the blurry halo scale, not the marker core - using its percentile
    band as the search range loses the small sigma (~2-3px) needed to
    resolve touching marker pairs (verified: merges known dimers into one
    detection). The fixed sigma set below was validated by hand against
    that failure mode instead."""
    best_sigmas = []
    for it in items:
        best = None
        for s in sigma_search:
            for val, y, x in log_peaks(it['inv'], it['region'], s, threshold_rel=0.3):
                if best is None or val > best[0]:
                    best = (val, s)
        if best is not None:
            best_sigmas.append(best[1])
    if best_sigmas:
        print(f'  (info) strongest-response sigma spread: '
              f'{np.percentile(best_sigmas, 20):.0f}-{np.percentile(best_sigmas, 80):.0f}')


def nms_peaks(candidates, min_dist, top_frac=0.35, max_n=4):
    """Greedy non-max suppression by response strength: rank all scale
    candidates, keep the strongest, then keep further ones only if they are
    both far enough from what's kept and still within top_frac of the best
    response (drops the long tail of weak noise peaks)."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: -c[0])
    best_val = candidates[0][0]
    kept = []
    for val, y, x, s in candidates:
        if val < top_frac * best_val:
            break
        if all(np.hypot(y - ky, x - kx) >= min_dist for _, ky, kx, _ in kept):
            kept.append((val, y, x, s))
        if len(kept) >= max_n:
            break
    return kept


def detect_in_crop(inv, region, sigmas, min_dist=6, top_frac=0.6, max_n=2):
    candidates = []
    for s in sigmas:
        for val, y, x in log_peaks(inv, region, s, threshold_rel=0.2):
            candidates.append((val, y, x, s))
    peaks = nms_peaks(candidates, min_dist, top_frac, max_n)
    return [(val, y, x, s * np.sqrt(2)) for val, y, x, s in peaks]


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


def make_cuts(result_dir, out_dir, margin=60, cols=8):
    """Crop each found instance out of its prediction image and save
    per-instance crops plus a per-image overview montage."""
    crops_root = os.path.join(result_dir, 'Crops')
    preds_dir = os.path.join(result_dir, 'preds')
    input_dir = os.path.join(result_dir, 'input')
    os.makedirs(out_dir, exist_ok=True)

    for img_id in sorted(os.listdir(crops_root)):
        img_path = os.path.join(preds_dir, img_id + '.png')
        input_path = os.path.join(input_dir, img_id + '.png')
        if not os.path.isfile(img_path) or not os.path.isfile(input_path):
            continue
        img = Image.open(img_path).convert('RGB')
        gray = np.array(Image.open(input_path).convert('L'))

        inst_dir = os.path.join(crops_root, img_id)
        inst_ids = sorted(os.listdir(inst_dir), key=int)

        cuts = []
        used_ids = []
        polys = []
        for idx in inst_ids:
            polyfiles = glob.glob(os.path.join(inst_dir, idx, 'polyRESC*.txt'))
            if not polyfiles:
                continue
            xs, ys, conf = read_polyRESC(polyfiles[0])
            x0, y0, x1, y1 = bbox(*img.size, xs, ys, margin)
            cuts.append(img.crop((x0, y0, x1, y1)))
            used_ids.append(idx)
            polys.append((xs, ys))

        if not cuts:
            continue

        img_out_dir = os.path.join(out_dir, img_id)
        os.makedirs(img_out_dir, exist_ok=True)
        for idx, cut, (xs, ys) in zip(used_ids, cuts, polys):
            cut.save(os.path.join(img_out_dir, f'{idx}.png'))

            mask = poly_mask(gray.shape, xs, ys)
            x0, y0, x1, y1 = bbox(*img.size, xs, ys, margin)
            save_inout_histogram(gray[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                                  os.path.join(img_out_dir, f'{idx}_hist.png'), idx)

        rows = int(np.ceil(len(cuts) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axs = np.array(axs).reshape(-1)
        for ax, cut, idx in zip(axs, cuts, used_ids):
            ax.imshow(cut)
            ax.set_title(idx, fontsize=8)
            ax.axis('off')
        for ax in axs[len(cuts):]:
            ax.axis('off')
        fig.suptitle(f'Image {img_id} - {len(cuts)} instances')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{img_id}_overview.png'), dpi=150)
        plt.close(fig)

        print(f'{img_id}: {len(cuts)} cuts -> {img_out_dir}')


def load_markers(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({'img_id': row['img_id'], 'crop_id': row['crop_id'],
                         'x': float(row['x_px']), 'y': float(row['y_px']), 'r': float(row['r_px'])})
    return rows


def auto_dimer_distance(rows, margin=1.15, default=90.0):
    """Same-crop marker pairs are real measured marker-marker distances (a
    cut with 2 markers almost certainly is a dimer) - use their spread to
    auto-set the neighbor-distance threshold instead of a hand-picked value."""
    by_crop = {}
    for r in rows:
        by_crop.setdefault((r['img_id'], r['crop_id']), []).append(r)

    dists = []
    for pts in by_crop.values():
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.hypot(pts[i]['x'] - pts[j]['x'], pts[i]['y'] - pts[j]['y']))

    if not dists:
        return default
    return float(np.percentile(dists, 75) * margin)


def classify_image(pts, distance, agg_size=3):
    """Direct port of the classification loop in
    automatic-dimer-analysis-tim/run_basic_analysis.py, applied to the
    markers pooled from all cuts of one image instead of a whole-image
    Hough pass."""
    n = len(pts)
    circles = np.zeros((1, n, 3))
    for i, p in enumerate(pts):
        circles[0, i] = [p['x'], p['y'], p['r']]

    reduced = find_closest_particles(circles)

    if agg_size >= 2 and n >= agg_size:
        agg_crit = find_ith_smallest_distance(circles, agg_size - 1)
        circles, reduced, agg_list, agg_pos = find_agglomerates(circles, reduced, distance, agg_crit)
    else:
        agg_list, agg_pos = [], []

    circles, reduced, mono1, mono_pos1 = find_monomer(circles, reduced, distance)
    circles, reduced, dim1, dim_pos1 = find_dimer(circles, reduced, distance)
    circles, reduced, mono2, mono_pos2 = find_monomer(circles, reduced, distance)
    circles, reduced, dim2, dim_pos2 = find_dimer(circles, reduced, distance)
    circles, reduced, dim3, dim_pos3 = check_for_equidistant_dimers(circles, reduced, distance)

    agg_arr = np.asarray(agg_list)
    dimer_arr = np.asarray(dim1 + dim2 + dim3)
    mono_arr = np.asarray(mono1 + mono2)
    used = agg_pos + mono_pos1 + dim_pos1 + mono_pos2 + dim_pos2 + dim_pos3
    unidentified = np.delete(circles, used, axis=1)[0]
    return agg_arr, dimer_arr, mono_arr, unidentified


def draw_classification(preds_path, out_path, agg_arr, dimer_arr, mono_arr, unidentified):
    img = np.array(Image.open(preds_path).convert('RGB'))
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)

    for row in agg_arr:
        ax.add_patch(plt.Circle((row[0], row[1]), row[2], color='m', fill=False))
    for row in mono_arr:
        ax.add_patch(plt.Circle((row[0], row[1]), row[2], color='r', fill=False))
    for row in dimer_arr:
        ax.add_patch(plt.Circle((row[0], row[1]), row[2], color='g', fill=False))
        ax.add_patch(plt.Circle((row[3], row[4]), row[5], color='g', fill=False))
        ax.plot([row[0], row[3]], [row[1], row[4]], color='g')
    for row in unidentified:
        ax.add_patch(plt.Circle((row[0], row[1]), row[2], color='b', fill=False))

    total = len(agg_arr) + len(mono_arr) + 2 * len(dimer_arr) + len(unidentified)
    if dimer_arr.shape[0] > 0:
        mean_dist = np.mean(dimer_arr[:, 6]) * NM_P_PX
        std_dist = np.std(dimer_arr[:, 6]) * NM_P_PX
        title = (f'{mean_dist:.2f} +/- {std_dist:.2f} nm, #{dimer_arr.shape[0]} dimers, '
                 f'#{mono_arr.shape[0]} monomers, #{agg_arr.shape[0]} agglomerated, '
                 f'#{unidentified.shape[0]} unidentified, #{total} total')
    else:
        title = (f'#0 dimers, #{mono_arr.shape[0]} monomers, #{agg_arr.shape[0]} agglomerated, '
                 f'#{unidentified.shape[0]} unidentified, #{total} total')
    ax.set_title(title)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_statistical_analysis(result_dir, cuts_dir, out_dir, agg_size=3):
    """Overlay marker counts on the preds images and reproduce the
    monomer/dimer/agglomerate statistics from automatic-dimer-analysis-tim,
    using markers.csv from find_markers_in_cuts as the particle list."""
    os.makedirs(out_dir, exist_ok=True)
    rows = load_markers(os.path.join(cuts_dir, 'markers.csv'))
    if not rows:
        print('No markers.csv to analyze.')
        return

    distance = auto_dimer_distance(rows)
    print(f'Auto-calibrated dimer/agglomerate distance threshold: {distance:.1f}px '
          f'({distance * NM_P_PX:.1f}nm)')

    by_img = {}
    for r in rows:
        by_img.setdefault(r['img_id'], []).append(r)

    agg_tot, dimer_tot, mono_tot, uniden_tot = [], [], [], []

    for img_id, pts in sorted(by_img.items()):
        preds_path = os.path.join(result_dir, 'preds', img_id + '.png')
        if not os.path.isfile(preds_path):
            continue

        agg_arr, dimer_arr, mono_arr, unidentified = classify_image(pts, distance, agg_size)
        draw_classification(preds_path, os.path.join(out_dir, f'{img_id}_classified.png'),
                             agg_arr, dimer_arr, mono_arr, unidentified)

        if agg_arr.shape[0]:
            agg_tot.append(agg_arr)
        if dimer_arr.shape[0]:
            dimer_tot.append(dimer_arr)
        if mono_arr.shape[0]:
            mono_tot.append(mono_arr)
        if unidentified.shape[0]:
            uniden_tot.append(unidentified)

        print(f'{img_id}: {dimer_arr.shape[0]} dimers, {mono_arr.shape[0]} monomers, '
              f'{agg_arr.shape[0]} agglomerated, {unidentified.shape[0]} unidentified')

    agg_tot = np.concatenate(agg_tot) if agg_tot else np.empty((0, 3))
    dimer_tot = np.concatenate(dimer_tot) if dimer_tot else np.empty((0, 7))
    mono_tot = np.concatenate(mono_tot) if mono_tot else np.empty((0, 3))
    uniden_tot = np.concatenate(uniden_tot) if uniden_tot else np.empty((0, 3))
    all_markers = np.array([[r['x'], r['y'], r['r']] for r in rows])

    if dimer_tot.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(12.8, 9.6))
        dist_nm = dimer_tot[:, 6] * NM_P_PX
        ax.hist(dist_nm, 50)
        ax.set_title(f'{np.mean(dist_nm):.2f} +/- {np.std(dist_nm):.2f} nm, '
                     f'#{dimer_tot.shape[0]} dimers, #{mono_tot.shape[0]} monomers, '
                     f'#{agg_tot.shape[0]} agglomerated, #{uniden_tot.shape[0]} unidentified')
        ax.set_xlabel('inter-marker distance [nm]')
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'total_distance_distribution.png'))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    r_nm = all_markers[:, 2] * NM_P_PX
    ax.hist(r_nm, 20)
    ax.set_title(f'{np.mean(r_nm):.2f} +/- {np.std(r_nm):.2f} nm, #{all_markers.shape[0]} markers')
    ax.set_xlabel('marker radius [nm]')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'marker_size_distribution.png'))
    plt.close(fig)

    with open(os.path.join(out_dir, 'all_markers_list.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_px', 'y_px', 'r_px', 'img_id'])
        writer.writerows([r['x'], r['y'], r['r'], r['img_id']] for r in rows)

    with open(os.path.join(out_dir, 'dimer_list.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1_px', 'y1_px', 'r1_px', 'x2_px', 'y2_px', 'r2_px', 'dist_px'])
        writer.writerows(dimer_tot.tolist())

    print(f'\nTotal: {dimer_tot.shape[0]} dimers, {mono_tot.shape[0]} monomers, '
          f'{agg_tot.shape[0]} agglomerated, {uniden_tot.shape[0]} unidentified, '
          f'{all_markers.shape[0]} markers -> {out_dir}')


if __name__ == '__main__':
    result_dir = r'F:\Data\DimerAnalysis\Result'
    out_dir = os.path.join(result_dir, 'cuts')
    analysis_dir = os.path.join(result_dir, 'analysis')
    marker_weights = None  # set to a trained best.pt path to use the YOLO detector
    make_cuts(result_dir, out_dir)
    find_markers_in_cuts(result_dir, out_dir, weights=marker_weights)
    run_statistical_analysis(result_dir, out_dir, analysis_dir)
