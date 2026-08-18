# Marker instance-segmentation design

Date: 2026-08-18

## Problem

`find_markers.py` finds gold-marker positions inside segmented DNA-molecule
crops with a classical multi-scale Laplacian-of-Gaussian peak detector
(`detect_in_crop` / `log_peaks`). This is a hand-tuned heuristic (sigma set,
NMS thresholds picked by hand against known failure cases). We want a learned
instance-segmentation model instead, matching how molecule detection itself
was bootstrapped (`IS_TrainData.py` -> YOLOv5-seg -> `predictTEM.py`).

## Scope

In scope:
- Synthetic training-data generator for marker instances (polygon labels,
  single class).
- YOLOv5-seg dataset yaml for the marker set.
- Inference function that runs a trained marker model on a crop and produces
  the same `(x, y, r)` row shape `markers.csv` already uses.
- Wiring `find_markers_in_cuts` to call the model-based detector instead of
  the LoG detector.
- Visual sanity check: synthetic samples vs. real crops.

Out of scope (explicitly deferred to the user):
- Running the actual `train.py` job (GPU time, hours).
- Real-data labeling/fine-tuning.

## Data generation

New file `GenMarkerTrainingData.py`, forked from `IS_TrainData.py`'s
`generate_image()` / `gen_DNA_shape()` (400x400 single-molecule tile — same
scale as the crops `make_cuts()` produces with `margin=60`). Molecule shape,
noise, and spot placement logic are reused unchanged.

Change: `generate_image()` currently sums every spot's Gaussian blob into one
shared array (`spot_arr`) and thresholds the sum into a single semantic mask
`spLbl` (class 2, no per-instance identity). The marker generator instead
keeps each spot's own blob mask before it's merged into the shared image,
so each spot keeps a separate identity:

1. For each spot `i`, render its Gaussian/cosine blob into a per-spot array
   (same math as today's inner loop), not into the shared `spot_arr`.
2. Threshold that array at the existing `spt_th` to get a per-spot binary
   mask.
3. Reject the spot if its mask area is too small (< ~9px, a degenerate
   speck) or empty (fully clipped off-canvas).
4. `cv2.findContours` + `cv2.approxPolyDP` (small epsilon) on the surviving
   mask -> polygon points.
5. Normalize by image size, write as a YOLO-seg instance line
   `0 x1 y1 x2 y2 ...` (class 0 = `Marker`).
6. Still add the per-spot blob into the shared `spot_arr` as before, so the
   rendered image (noise + molecule + spots) is pixel-identical to today's
   generator — only the label extraction changes.

Output layout mirrors the existing molecule set:
```
SynthData/MarkerSet1/images/train/000000.png ...
SynthData/MarkerSet1/labels/train/000000.txt ...
SynthData/MarkerSet1/images/val/...
SynthData/MarkerSet1/labels/val/...
```
Images with zero surviving marker instances still get written (empty label
file) — needed so the model also learns what "no marker" looks like inside
molecule background/noise.

## Dataset yaml

`yolov5-master/yolov5-master/data/SetMarkerIS.yaml`:
```yaml
path: <SynthData/MarkerSet1 root>
train: images/train
val: images/val
names:
  0: Marker
```

## Visualization

Small script (or a function in `GenMarkerTrainingData.py` run under
`if __name__ == '__main__'`) that generates ~20 samples, draws the marker
polygons on top (reusing the `cv2.fillPoly`-with-alpha pattern from
`LabelConversion.viz_polygon`), and saves them next to a few real crops
pulled from `F:\Data\DimerAnalysis\Result\cuts` for a side-by-side visual
check. No automated correctness test — there is no ground truth for real
images beyond "does this look like the real markers."

## Training

Not automated. Documented command for the user to run themselves:
```
python yolov5-master/yolov5-master/segment/train.py \
  --data yolov5-master/yolov5-master/data/SetMarkerIS.yaml \
  --weights yolov5s-seg.pt --img 400 --epochs <N>
```

## Inference integration

New `detect_markers_yolo(crop_rgb, model, device, imgsz, conf_thres,
iou_thres)` in `find_markers.py`, modeled on the `run()` function at the top
of `predictTEM.py` (the clean reusable entry point — not the experimental
`apply_image*` functions elsewhere in that file): load once via
`DetectMultiBackend`, run inference + `process_mask` + `masks2segments` per
crop, take each predicted polygon's centroid as `(x, y)` and
`sqrt(area/pi)` as `r`.

`find_markers_in_cuts(result_dir, out_dir, margin=25, weights=None)`: when
`weights` is given, loads the model once and calls `detect_markers_yolo` per
crop instead of `detect_in_crop`/`log_peaks`. When `weights` is `None`,
keeps today's LoG behavior (no trained model exists yet this session, so the
old path stays as the default until a model is trained). Output rows/columns
in `markers.csv` are unchanged either way, so `run_statistical_analysis`
needs no changes.

## Testing

- Visual check of synthetic samples vs. real crops (above).
- Once a model is trained, spot-check `detect_markers_yolo` output on a
  handful of real crops from `F:\Data\DimerAnalysis\Result`, comparing
  against the existing LoG detector's output on the same crops.
