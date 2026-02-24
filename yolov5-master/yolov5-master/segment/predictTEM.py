
import argparse
import copy
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from multiprocessing import Process, Pool
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from shapely.geometry import Polygon, Point
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import json
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
import PreprocessImages
import pandas as pd

THREADS = 15

NM_P_PX = 0.8431 # 0.697

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))
                    ]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                    / 255
                    if retina_masks
                    else im[i],
                )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():

    weights = r'D:\seifert\PycharmProjects\LiuDimer\yolov5-master\yolov5-master\runs\train-seg\exp9\weights\best.pt'
    weights = r'D:\seifert\PycharmProjects\LiuDimer\yolov5-master\yolov5-master\runs\train-seg\exp10\weights\last.pt'

    imgs_in =  r'D:\seifert\PycharmProjects\LiuDimer\SynthData\Test_Real\Images'
    imgs_out = r'D:\seifert\PycharmProjects\LiuDimer\SynthData\Test_Real\YOLOV2_400'
    os.makedirs(imgs_out, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=weights, help="model path(s)")
    parser.add_argument("--source", type=str, default=imgs_in, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[400], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=imgs_out, help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=1, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt




def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


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

def calc_poly_area(polygon):
    real_pts_x = []
    real_pts_y = []
    for i, x in enumerate(polygon):
        if i % 2 == 0:
            real_pts_x.append(x)
        else:
            real_pts_y.append(x)
    real_pts_x = np.array(real_pts_x)
    real_pts_y = np.array(real_pts_y)
    area = 0.5*np.abs(np.dot(real_pts_x,np.roll(real_pts_y,1))-np.dot(real_pts_y,np.roll(real_pts_x,1)))
    return area


def viz_polygon(im, lb, resf):

    with open(lb, 'r') as f:
        labels = f.read().splitlines()
    img = cv2.imread(im)
    h, w = img.shape[:2]
    for label in labels:
        class_id, *poly = label.split(' ')
        # Reshape function
        xs = []
        ys = []
        poly = [float(x) for x in poly]
        for i, elem in enumerate(poly):
            if i % 2 == 0:
                xs.append(elem)
            else:
                ys.append(elem)
        pts = []
        for x, y in zip(xs, ys):
            v = np.array([x, y])
            pts.append(v)
        # pts = sorted(pts, key=clockwiseangle_and_distance)
        polyres = []
        for p in pts:
            polyres.append(p[0])
            polyres.append(p[1])
        poly = polyres
        poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
        poly *= [w, h]  # Unscale
        c = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        cv2.polylines(img, [poly.astype('int')], True, c,
                      2)  # Draw Poly Lines
        cv2.fillPoly(img, [poly.astype('int')], c, cv2.LINE_AA) # Draw area
        if max(img.shape[0], img.shape[1]) < 400:
            img = cv2.resize(img, (600, 600))
        cv2.imwrite(resf, img)

def viz_all_polygons_NonInv(im, lblfiles, resf):
    img = cv2.imread(im)
    img = 255 - img
    # print(img)
    h, w = img.shape[:2]
    color = (0, 0, 255)
    # assert 1 == 2
    overlay = copy.deepcopy(img)
    for lb in lblfiles:
        with open(lb, 'r') as f:
            labels = f.read().splitlines()
        for label in labels:
            class_id, *poly = label.split(' ')
            # Reshape function
            xs = []
            ys = []
            poly = [float(x) for x in poly]
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    xs.append(elem)
                else:
                    ys.append(elem)
            pts = []
            for x, y in zip(xs, ys):
                v = np.array([x, y])
                pts.append(v)
            # pts = sorted(pts, key=clockwiseangle_and_distance)
            polyres = []
            for p in pts:
                polyres.append(p[0])
                polyres.append(p[1])
            poly = polyres
            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
            poly *= [w, h]  # Unscale

            cv2.polylines(overlay, [poly.astype('int')], True, color, 2)  # Draw Poly Lines

            # cv2.fillPoly(img, [poly.astype('int')], c, cv2.LINE_AA) # Draw area

            cv2.fillPoly(overlay, [poly.astype('int')], color, cv2.LINE_AA)  # Draw area


        # plt.switch_backend('TkAgg')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

       # plt.imshow(img)
       # plt.title('Img')
       # plt.show()
       # plt.imshow(overlay)
       # plt.title('OVL')
       # plt.show()


    img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    #plt.imshow(img)
    #plt.title('Img')
    # plt.show()
    if max(img.shape[0], img.shape[1]) < 400:
        img = cv2.resize(img, (600, 600))
    cv2.imwrite(resf, img)

def apply_image(imagef, modelf, conf_thres=0.25, iou_thres=0.45):
    save_dir = Path('Results')
    tempf = os.path.join(save_dir, 'Crops')
    os.makedirs(tempf, exist_ok=True)



    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(imagef)
    arr = np.array(img)
    # img.show('Input')
    visualize=False

    arr = arr[:2048, :]

    ss = 1

    arr = arr[::ss, ::ss]

    ks = 101
    mid = 51
    sig = 40 / ss
    kernel = np.zeros((ks, ks))
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x, y] = np.exp(- np.sqrt((x - mid) ** 2 + (y - mid) ** 2) / sig)

    kernel /= np.sum(kernel)
    # plt.imshow(kernel)
    # plt.title('Large Kernel')
    # plt.show()
    arr = arr.astype(float)
    plt.imshow(arr)
    plt.title('Plain')
    plt.show()

    arr -= np.median(arr, axis=0, keepdims=True)
    plt.imshow(arr)
    plt.title('Ax0')
    plt.show()
    arr -= np.median(arr, axis=1, keepdims=True)
    plt.imshow(arr)
    plt.title('Ax1')
    plt.show()

    # arr = cv2.filter2D(arr, -1, kernel)

    plt.imshow(arr)
    plt.title('Conv')
    plt.show()

    height = img.size[1]
    width = img.size[0]

    plt.imshow(arr)
    plt.show()

    arr_div = np.abs(arr - np.median(arr))
    plt.imshow(arr_div)
    plt.show()

    arr_div = cv2.filter2D(arr_div, -1, kernel)

    arr_div -= np.median(arr_div)


    plt.imshow(arr_div)
    plt.title('CNV')
    plt.show()



    th = 1

    arr_bin = np.zeros_like(arr_div, dtype=int)
    arr_bin[arr_div > th] = 1

    plt.imshow(arr_bin)
    plt.show()

    cls = hoshen_koppelmann(arr_bin, minsize=10)

    plt.imshow(cls)
    plt.show()

    centers = []

    for vals in np.unique(cls):
        if vals < 0:
            continue

        target_indices  = np.where(cls == vals)
        avg_row = np.mean(target_indices[0])
        avg_col = np.mean(target_indices[1])
        centers.append(np.array([avg_row, avg_col]))

    # Filter Positions

    min_ds = np.inf * np.ones(len(centers))
    for i, center in enumerate(centers):
        for j, ot in enumerate(center):
            if not i == j:
                min_ds[i] = min(min_ds[i], np.linalg.norm(center - ot))



    size = 400
    for center in centers:

        min0 = max(int(center[0] - size/2), 0)
        max0 = min0 + size
        if max0 >= arr.shape[0]:
            max0 = arr.shape[0] - 1
            min0 = max0 - size

        min1 = max(int(center[1] - size / 2), 0)
        max1 = min1 + size
        if max1 >= arr.shape[1]:
            max1 = arr.shape[1] - 1
            min1 = max1 - size

        spt = arr[min0:max0, min1:max1]
        plt.imshow(spt, cmap='gray')
        plt.title(str(spt.shape))
        plt.show()





    assert 5 == 6





    source = str(imagef)
    save_img = True  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Directories
    Path(os.path.join(save_dir,"labels")).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device()
    # print('DEVICE: ', device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img.size, s=stride)  # check image size
    vid_stride = 1

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    view_img = True

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(os.path.join(save_dir, Path(path).stem), mkdir=True) if visualize else False
            pred, proto = model(im, augment=False, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0  # for save_crop
            annotator = Annotator(im0, line_width=1, example=str(names))
            if len(det):

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                segments = [
                    scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))
                ]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=im[i],
                )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf)
                    with open(f"{txt_path}.txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def apply_imageENTIRE(imagef, weights, resultf=None, conf_thres=0.25, iou_thres=0.45, plain=True,img_idx = 0, show_indiv=False):
    save_dir = Path('Results') if resultf is None else resultf
    tempf = os.path.join(save_dir, 'Crops')
    os.makedirs(tempf, exist_ok=True)
    img = Image.open(imagef)

    visualize = False


    arr = np.array(img).astype(float)
    arr = arr[:2048, :]
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]

    arr = 255 - arr

    if not plain:
        # plt.imshow(arr, cmap='gray')
        # plt.show()

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
        arr /= np.amax(arr)

        # plt.imshow(arr)
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
    plt.imsave(os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png'), arr, cmap='gray', vmin=0, vmax=1)
    infile_file = os.path.join(save_dir, f'{str(img_idx).zfill(4)}_in.png')
    shutil.copy(os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png'), infile_file)
    imagefn = os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png')

    image_name = os.path.basename(imagefn).split('.')[0]



    source = str(imagefn)
    save_img = True  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Directories
    Path(os.path.join(save_dir,"labels")).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device()
    # print('DEVICE: ', device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size([arr.shape[0], arr.shape[1]], s=stride)  # check image size

    print("ImGSZ", imgsz)
    vid_stride = 1

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    view_img = True

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(os.path.join(save_dir, Path(path).stem), mkdir=True) if visualize else False
            pred, proto = model(im, augment=False, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0  # for save_crop
            annotator = Annotator(im0, line_width=1, example=str(names))
            if len(det):

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                segments = [
                    scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))
                ]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=im[i],
                )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf)
                    with open(f"{txt_path}.txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

            # Stream results
            im0 = annotator.result()
            if False:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    crops = {}


    idx = 0
    centers = []
    if not os.path.isfile(os.path.join(save_dir, 'labels', image_name + '.txt')):
        return  {}
    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        for line in f:
            parts = line.split(' ')
            poly = [float(x) for x in parts[1:-1]]
            w = arr.shape[1]
            h = arr.shape[0]
            xs = []
            ys = []
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    xs.append(w * elem)
                else:
                    ys.append(h * elem)

            centers.append(np.array([(max(xs) + min(xs))/2, (max(ys) + min(ys)) / 2]))


    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        os.makedirs(os.path.join(save_dir, 'Crops', image_name), exist_ok=True)

        for line in f:
            crops[idx] = {}
            crpfld = os.path.join(save_dir, 'Crops', image_name, str(idx))
            os.makedirs(crpfld, exist_ok=True)

            parts = line.split(' ')
            conf = float(parts[-1])
            cat = parts[1]
            poly = [float(x) for x in parts[1:-1]]

            crops[idx]['cat'] = cat
            crops[idx]['conf'] = conf
            crops[idx]['fld'] = crpfld
            with open(os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(line)

            with open(os.path.join(crpfld, f'conf{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(str(conf))

            if show_indiv:
                assert 2 == 4
                viz_polygon(infile_file, os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'), os.path.join(crpfld, f'poly{str(idx).zfill(3)}.png'))


            poly_resc = []
            w = arr.shape[1]
            h = arr.shape[0]
            xs = []
            ys = []
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    poly_resc.append(w * elem)
                    xs.append(w * elem)
                else:
                    poly_resc.append(h * elem)
                    ys.append(h * elem)

            center = np.array([(max(xs) + min(xs))/2, (max(ys) + min(ys)) / 2])
            crops[idx]['center_x'] = center[0] * NM_P_PX
            crops[idx]['center_y'] = center[1] * NM_P_PX

            dists = []
            for c in centers:
                d = np.linalg.norm(center - c) * NM_P_PX
                if d > 0:
                    dists.append(d)
            dists = sorted(dists)

            while(len(dists)) < 5:
                dists.append(max(w, h) * NM_P_PX)



            with open(os.path.join(crpfld, f'polyRESC{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{cat} {' '.join([str(x) for x in poly_resc])} {conf}\n")

            with open(os.path.join(crpfld, f'polyRESC{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{cat} {' '.join([str(x) for x in poly_resc])} {conf}\n")

            area = calc_poly_area(poly_resc)
            area_nm = area * NM_P_PX**2

            with open(os.path.join(crpfld, f'area{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{area }px\n{area_nm} nm\n")

            crops[idx]['area'] = area
            crops[idx]['area_nm'] = area_nm
            crops[idx]['w'] = w
            crops[idx]['h'] = h
            crops[idx]['nn1'] = dists[0]
            crops[idx]['nn2'] = dists[1]
            crops[idx]['nn3'] = dists[2]
            crops[idx]['nn4'] = dists[3]
            crops[idx]['nn5'] = dists[4]

            idx += 1

    return crops

def cinv(pr):
    fnin, fnout = pr
    arr = np.array(Image.open(fnin))
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]
    arr = arr[:2048, :]
    # plt.imshow(arr)
    # plt.show()
    arr = 255 - arr
    plt.imsave(fnout, arr, cmap='gray')


def iou(mat1, mat2):
    i = np.sum(np.multiply(mat1, mat2))
    u = np.sum(np.maximum(mat1, mat2))
    return i / u

def pol_coos(x, y, theta, new_mid_x, new_mid_y, old_mid_x, old_mid_y, newdim,res):
    dx = x - new_mid_x
    dy = new_mid_y - y
    # print('DX, DY', dx, dy)
    v = np.array([dx, dy])
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    vs = np.dot(v, rot)
    # print("V: ", vs)
    # print(f'{(newdim / res)} * {vs[0]} + {old_mid_x}')
    old_x = (newdim / res) * vs[0] + old_mid_x
    old_y = (newdim / res) * vs[1] + old_mid_y
    # print("X, Y", x, y, ' --> ', old_x, old_y)
    return old_x, old_y

def gen_shape(poly, theta, new_mid_x, new_mid_y, old_mid_x, old_mid_y, newdim,res):
    mat = np.zeros((res, res))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ox, oy = pol_coos(j, i, theta, new_mid_x, new_mid_y, old_mid_x, old_mid_y, newdim,res)
            if poly.contains(Point(ox, oy)):
                mat[i, j] = 1
    return mat

def rect(xmin, xmax, ymin, ymax, res):
    arr = np.zeros((res, res))
    arr[ymin:ymax, xmin:xmax] = 1
    return arr

def approximate_poly_as_flat_rect(polygon, res=100, angle_res=100, iterations=2, savefld=None):
    angle = 0
    start = time.perf_counter()

    xs = []
    ys = []
    # print('Polygon', polygon)
    for pair in polygon:
        xs.append(pair[0])
        ys.append(pair[1])

    # print('XS: ', xs)
    # print('YS: ', ys)

    xmax = np.amax(xs)
    xmin = np.amin(xs)
    ymax = np.amax(ys)
    ymin = np.amin(ys)
    wmax = np.sqrt(2) * (xmax - xmin)
    hmax = np.sqrt(2) * (ymax - ymin)
    newdim = max(wmax, hmax)

    old_mid_x = (xmax + xmin) / 2
    old_mid_y = (ymax + ymin) / 2

    # print(xmin, xmax, ymin, ymax, wmax, hmax, old_mid_x, old_mid_y)

    new_mid_x = (res - 1) / 2
    new_mid_y = (res - 1) / 2

    max_iou_angles = []
    best_pair = (0, 0, res- 1, 0, res-1)
    best_iou = -1
    plg = Polygon(polygon)
    # print('PLG: ', plg)
    # plt.switch_backend('TkAgg')
    arr = gen_shape(plg, theta=angle, new_mid_x=new_mid_x, new_mid_y=new_mid_y, old_mid_x=old_mid_x, old_mid_y=old_mid_y, newdim=newdim,res=res)
    # plt.imshow(arr)
    # plt.show()
    xmin = 0
    xmax = arr.shape[1] - 1
    ymin = 0
    ymax = arr.shape[0] - 1
    for iter in range(iterations):
        xmins = []
        iou_xmin = []
        for i in range(int(0.8 * arr.shape[1])):
            xmins.append(i)
            arr2 = rect(i, xmax, ymin, ymax, res)
            iou_xmin.append(iou(arr, arr2))
        plt.plot(xmins, iou_xmin)
        newxmin = xmins[np.argmax(iou_xmin)]
        # plt.title(f'Xmin: {newxmin}')
        # plt.show()
        # if iter > 0 and xmin != newxmin:
        #     print(f'Xmin: {xmin} --> {newxmin}')
        xmin = newxmin
        ymins = []
        iou_ymin = []
        for i in range(int(0.8 * arr.shape[0])):
            ymins.append(i)
            arr2 = rect(xmin, xmax, i, ymax, res)
            iou_ymin.append(iou(arr, arr2))
        # plt.plot(ymins, iou_ymin)
        newymin = ymins[np.argmax(iou_ymin)]
        # plt.title(f'Ymin: {newymin}')
        # plt.show()
        # if iter > 0 and ymin != newymin:
        #     print(f'Ymin: {ymin} --> {newymin}')
        ymin = newymin
        xmaxs = []
        iou_xmax = []
        for i in range(xmin, arr.shape[1] - 1):
            xmaxs.append(i)
            arr2 = rect(xmin, i, ymin, ymax, res)
            iou_xmax.append(iou(arr, arr2))
        # plt.plot(xmaxs, iou_xmax)
        newxmax = xmaxs[np.argmax(iou_xmax)]
        # plt.title(f'Xmax: {newxmax}')
        # plt.show()
        # if iter > 0 and xmax != newxmax:
        #     print(f'Xmax: {xmax} --> {newxmax}')
        xmax = newxmax
        ymaxs = []
        iou_ymax = []
        for i in range(ymin, arr.shape[0] - 1):
            ymaxs.append(i)
            arr2 = rect(xmin, xmax, ymin, i, res)
            iou_ymax.append(iou(arr, arr2))
        # plt.plot(ymaxs, iou_ymax)
        newymax = ymaxs[np.argmax(iou_ymax)]
        # plt.title(f'Ymax: {newymax}')
        # plt.show()
        # if iter > 0 and ymax != newymax:
        #     print(f'Ymax: {ymax} --> {newymax}')
        ymax = newymax
    arrRes = rect(xmin, xmax, ymin, ymax, res)
    iouRes = iou(arr, arrRes)
    max_iou_angles.append(iouRes)
    if iouRes > best_iou:
        best_iou = iouRes
        best_pair = (angle, xmin, xmax, ymin, ymax)


    new_xrect = best_pair[2] - best_pair[1]
    new_yrect = best_pair[4] - best_pair[3]

    old_xrect = new_xrect * res / newdim
    old_yrect = new_yrect * res / newdim

    old_xrect = new_xrect * newdim / res
    old_yrect = new_yrect * newdim / res

    if old_xrect > old_yrect:
        angle = best_pair[0] * 180 / np.pi
        width = old_xrect
        height = old_yrect
    else:
        angle = (best_pair[0] * 180 / np.pi) + 90
        width = old_yrect
        height = old_yrect

    if savefld is not None:
        best_angle_arr = gen_shape(plg, theta=best_pair[0], new_mid_x=new_mid_x, new_mid_y=new_mid_y, old_mid_x=old_mid_x, old_mid_y=old_mid_y, newdim=newdim,res=res)
        best_angle_rect = rect(best_pair[1], best_pair[2], best_pair[3], best_pair[4], res)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(xs, ys)
        axs[0, 1].imshow(best_angle_arr)
        axs[1, 0].imshow(best_angle_rect)
        kpx = [best_pair[1], best_pair[1], best_pair[2], best_pair[2]]
        kpy = [best_pair[3], best_pair[4], best_pair[3], best_pair[4]]
        axs[1, 1].imshow(best_angle_arr)
        axs[1, 1].scatter(kpx, kpy)
        plt.suptitle(f'IoU: {best_iou:.4f}')
        plt.show()
        plt.savefig(os.path.join(savefld, 'rect.png'))
        plt.close(fig)

    dur = time.perf_counter() - start
    # print(f"{dur}s @ {res}x{angle_res}x{iterations}")

    return angle, width, height, best_iou
def approximate_poly_as_rect(polygon, res=100, angle_res=100, iterations=2, savefld=None):
    start = time.perf_counter()
    angles = np.linspace(0, np.pi/2, angle_res)
    d_angle = angles[1] - angles[0]

    xs = []
    ys = []
    # print('Polygon', polygon)
    for pair in polygon:
        xs.append(pair[0])
        ys.append(pair[1])

    # print('XS: ', xs)
    # print('YS: ', ys)

    xmax = np.amax(xs)
    xmin = np.amin(xs)
    ymax = np.amax(ys)
    ymin = np.amin(ys)
    wmax = np.sqrt(2) * (xmax - xmin)
    hmax = np.sqrt(2) * (ymax - ymin)
    newdim = max(wmax, hmax)

    old_mid_x = (xmax + xmin) / 2
    old_mid_y = (ymax + ymin) / 2

    # print(xmin, xmax, ymin, ymax, wmax, hmax, old_mid_x, old_mid_y)

    new_mid_x = (res - 1) / 2
    new_mid_y = (res - 1) / 2

    max_iou_angles = []
    best_pair = (0, 0, res- 1, 0, res-1)
    best_iou = -1
    plg = Polygon(polygon)
    # print('PLG: ', plg)
    plt.switch_backend('TkAgg')
    for angle in tqdm(angles, desc='Approxi Theta', disable=True, position=0, leave=True):
        arr = gen_shape(plg, theta=angle, new_mid_x=new_mid_x, new_mid_y=new_mid_y, old_mid_x=old_mid_x, old_mid_y=old_mid_y, newdim=newdim,res=res)
        # plt.imshow(arr)
        # plt.show()
        xmin = 0
        xmax = arr.shape[1] - 1
        ymin = 0
        ymax = arr.shape[0] - 1

        for iter in range(iterations):

            xmins = []
            iou_xmin = []

            for i in range(int(0.8 * arr.shape[1])):
                xmins.append(i)
                arr2 = rect(i, xmax, ymin, ymax, res)
                iou_xmin.append(iou(arr, arr2))

            #plt.plot(xmins, iou_xmin)
            newxmin = xmins[np.argmax(iou_xmin)]
            #plt.title(f'Xmin: {newxmin}')
            #plt.show()
            # if iter > 0 and xmin != newxmin:
            #     print(f'Xmin: {xmin} --> {newxmin}')
            xmin = newxmin

            ymins = []
            iou_ymin = []

            for i in range(int(0.8 * arr.shape[0])):
                ymins.append(i)
                arr2 = rect(xmin, xmax, i, ymax, res)
                iou_ymin.append(iou(arr, arr2))

            #plt.plot(ymins, iou_ymin)
            newymin = ymins[np.argmax(iou_ymin)]
            #plt.title(f'Ymin: {newymin}')
            #plt.show()
            # if iter > 0 and ymin != newymin:
            #     print(f'Ymin: {ymin} --> {newymin}')
            ymin = newymin

            xmaxs = []
            iou_xmax = []

            for i in range(xmin, arr.shape[1] - 1):
                xmaxs.append(i)
                arr2 = rect(xmin, i, ymin, ymax, res)
                iou_xmax.append(iou(arr, arr2))

            #plt.plot(xmaxs, iou_xmax)
            newxmax = xmaxs[np.argmax(iou_xmax)]
            #plt.title(f'Xmax: {newxmax}')
            #plt.show()
            # if iter > 0 and xmax != newxmax:
            #     print(f'Xmax: {xmax} --> {newxmax}')
            xmax = newxmax

            ymaxs = []
            iou_ymax = []

            for i in range(ymin, arr.shape[0] - 1):
                ymaxs.append(i)
                arr2 = rect(xmin, xmax, ymin, i, res)
                iou_ymax.append(iou(arr, arr2))

            #plt.plot(ymaxs, iou_ymax)
            newymax = ymaxs[np.argmax(iou_ymax)]
            #plt.title(f'Ymax: {newymax}')
            #plt.show()
            # if iter > 0 and ymax != newymax:
            #     print(f'Ymax: {ymax} --> {newymax}')
            ymax = newymax

        arrRes = rect(xmin, xmax, ymin, ymax, res)
        iouRes = iou(arr, arrRes)
        max_iou_angles.append(iouRes)

        if iouRes > best_iou:
            best_iou = iouRes
            best_pair = (angle, xmin, xmax, ymin, ymax)

    angles_fine = np.linspace(best_pair[0] - d_angle, best_pair[0] + d_angle, angle_res)
    for angle in tqdm(angles_fine, desc='Approxi Theta', disable=True, position=0, leave=True):
        arr = gen_shape(plg, theta=angle, new_mid_x=new_mid_x, new_mid_y=new_mid_y, old_mid_x=old_mid_x, old_mid_y=old_mid_y, newdim=newdim,res=res)
        _, xmin, xmax, ymin, ymax = best_pair
        for iter in range(iterations):

            xmins = []
            iou_xmin = []
            xmintemp_min = int(max(xmin-res/10, 0))
            xmintemp_max = int(min(1+xmin+res/10, arr.shape[1]))

            for i in range(xmintemp_min, xmintemp_max):
                xmins.append(i)
                arr2 = rect(i, xmax, ymin, ymax, res)
                iou_xmin.append(iou(arr, arr2))

            #plt.plot(xmins, iou_xmin)
            newxmin = xmins[np.argmax(iou_xmin)]
            #plt.title(f'Xmin: {newxmin}')
            #plt.show()
            # if iter > 0 and xmin != newxmin:
            #     print(f'Xmin: {xmin} --> {newxmin}')
            xmin = newxmin

            ymins = []
            iou_ymin = []

            ymintemp_min = int(max(ymin - res / 10, 0))
            ymintemp_max = int(min(1 + ymin + res / 10, arr.shape[0]))

            for i in range(ymintemp_min, ymintemp_max):
                ymins.append(i)
                arr2 = rect(xmin, xmax, i, ymax, res)
                iou_ymin.append(iou(arr, arr2))

            #plt.plot(ymins, iou_ymin)
            newymin = ymins[np.argmax(iou_ymin)]
            #plt.title(f'Ymin: {newymin}')
            #plt.show()
            # if iter > 0 and ymin != newymin:
            #     print(f'Ymin: {ymin} --> {newymin}')
            ymin = newymin

            xmaxs = []
            iou_xmax = []
            xmaxtemp_min = int(max(xmax - res / 10, 0))
            xmaxtemp_max = int(min(1 + xmax + res / 10, arr.shape[1]))

            for i in range(xmaxtemp_min, xmaxtemp_max):
                xmaxs.append(i)
                arr2 = rect(xmin, i, ymin, ymax, res)
                iou_xmax.append(iou(arr, arr2))

            #plt.plot(xmaxs, iou_xmax)
            newxmax = xmaxs[np.argmax(iou_xmax)]
            #plt.title(f'Xmax: {newxmax}')
            #plt.show()
            # if iter > 0 and xmax != newxmax:
            #     print(f'Xmax: {xmax} --> {newxmax}')
            xmax = newxmax

            ymaxs = []
            iou_ymax = []
            ymaxtemp_min = int(max(ymax - res / 10, 0))
            ymaxtemp_max = int(min(1 + ymax + res / 10, arr.shape[0]))

            for i in range(ymaxtemp_min, ymaxtemp_max):
                ymaxs.append(i)
                arr2 = rect(xmin, xmax, ymin, i, res)
                iou_ymax.append(iou(arr, arr2))

            #plt.plot(ymaxs, iou_ymax)
            newymax = ymaxs[np.argmax(iou_ymax)]
            #plt.title(f'Ymax: {newymax}')
            #plt.show()
            # if iter > 0 and ymax != newymax:
            #     print(f'Ymax: {ymax} --> {newymax}')
            ymax = newymax

        arrRes = rect(xmin, xmax, ymin, ymax, res)
        iouRes = iou(arr, arrRes)
        max_iou_angles.append(iouRes)

        if iouRes > best_iou:
            best_iou = iouRes
            best_pair = (angle, xmin, xmax, ymin, ymax)


    new_xrect = best_pair[2] - best_pair[1]
    new_yrect = best_pair[4] - best_pair[3]

    old_xrect = new_xrect * res / newdim
    old_yrect = new_yrect * res / newdim

    old_xrect = new_xrect * newdim / res
    old_yrect = new_yrect * newdim / res

    if old_xrect > old_yrect:
        angle = best_pair[0] * 180 / np.pi
        width = old_xrect
        height = old_yrect
    else:
        angle = (best_pair[0] * 180 / np.pi) + 90
        width = old_yrect
        height = old_yrect

    if savefld is not None:
        best_angle_arr = gen_shape(plg, theta=best_pair[0], new_mid_x=new_mid_x, new_mid_y=new_mid_y, old_mid_x=old_mid_x, old_mid_y=old_mid_y, newdim=newdim,res=res)
        best_angle_rect = rect(best_pair[1], best_pair[2], best_pair[3], best_pair[4], res)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(xs, ys)
        axs[0, 1].imshow(best_angle_arr)
        axs[1, 0].imshow(best_angle_rect)
        kpx = [best_pair[1], best_pair[1], best_pair[2], best_pair[2]]
        kpy = [best_pair[3], best_pair[4], best_pair[3], best_pair[4]]
        axs[1, 1].imshow(best_angle_arr)
        axs[1, 1].scatter(kpx, kpy)
        plt.suptitle(f'IoU: {best_iou:.4f}')
        plt.savefig(os.path.join(savefld, 'rect.png'))
        plt.close(fig)

    dur = time.perf_counter() - start
    # print(f"{dur}s @ {res}x{angle_res}x{iterations}")

    return angle, width, height, best_iou

def visualize_preds(pair):
    print("VP")
    resf, save_dir = pair
    image_name = resf.split('.')[0]
    arr = np.array(Image.open(os.path.join(save_dir, 'input', resf)))
    loc_crop = {}
    idx = 0
    centers = []


    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        for line in f:
            parts = line.split(' ')
            poly = [float(x) for x in parts[1:-1]]
            w = arr.shape[1]
            h = arr.shape[0]
            xs = []
            ys = []
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    xs.append(w * elem)
                else:
                    ys.append(h * elem)

            centers.append(np.array([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2]))

    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        os.makedirs(os.path.join(save_dir, 'Crops', image_name), exist_ok=True)

        plgs = []

        for line in f:
            loc_crop[idx] = {}
            crpfld = os.path.join(save_dir, 'Crops', image_name, str(idx))
            os.makedirs(crpfld, exist_ok=True)

            parts = line.split(' ')
            conf = float(parts[-1])
            cat = parts[0]
            poly = [float(x) for x in parts[1:-1]]

            loc_crop[idx]['cat'] = cat
            loc_crop[idx]['conf'] = conf
            loc_crop[idx]['fld'] = crpfld
            with open(os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(line)

            with open(os.path.join(crpfld, f'conf{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(str(conf))

            plgs.append(os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'))
            idx += 1

        os.makedirs(os.path.join(save_dir, 'NonInvPreds'), exist_ok=True)
        viz_all_polygons_NonInv(os.path.join(save_dir, 'input', resf), plgs, os.path.join(save_dir, 'NonInvPreds', f'{image_name}.png'))


def analyze_result(pair):
    show_indiv = False

    def filter_pairs(pairs, th=2, minf=100):
        maxi = np.amax([e[1] for e in pairs if e[0] > minf])
        pairs_filetred = [p for p in pairs if p[0] > minf and p[1] > maxi / th]
        prs = pairs_filetred
        pairs_filetred2 = []
        ds = []
        for i1, pair in enumerate(prs):
            for i2, p2 in enumerate(prs):
                shrt = False
                if i1 != i2:
                    if abs(pair[0] - p2[0]) < 25:
                        shrt = True

            if not shrt:
                pairs_filetred2.append(pair)

        prs = pairs_filetred2
        xs = [x[0] for x in prs]
        ys = [x[1] for x in prs]
        plt.scatter(xs, ys)
        iterations = []
        key = prs[0]
        iterations.append((key, 1))
        for i, pair in enumerate(prs):
            if i == 0:
                continue
            ober = pair[0] / key[0]
            if abs(round(ober) - ober) < 0.1:
                iterations.append((pair, round(ober)))

        freqs = []
        for i in iterations:
            f = i[0][0] / i[1]
            freqs.append(f)
        # print(freqs)
        return np.average(freqs)

    def find_keypoint(x_vals, x_confs, fx):
        rng = np.linspace(min(x_vals), min(x_vals) + fx, 1000)
        sums = []
        for x_start in tqdm(rng, disable=True):
            sum = 0
            for x2, c2 in zip(x_vals, x_confs):
                d = abs(x_start - x2)
                per = abs((x_start + (round(d / fx) * fx)) - x2)
                sum += c2 * per

            sums.append(sum)
        plt.plot(rng, sums)
        return rng[np.argmin(sums)]

    resf, save_dir = pair
    image_name = resf.split('.')[0]
    out_json = os.path.join(save_dir, 'json_res', image_name + '.json')

    if os.path.isfile(out_json):
        return
    arr = np.array(Image.open(os.path.join(save_dir, 'input', resf)))
    loc_crop = {}
    idx = 0
    centers = []

    if not os.path.isfile(os.path.join(save_dir, 'labels', image_name + '.txt')):
        with open(out_json, 'w') as f:
            json.dump({}, f)
            return

    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        for line in f:
            parts = line.split(' ')
            if len(parts) < 4:
                continue
            poly = [float(x) for x in parts[1:-1]]
            w = arr.shape[1]
            h = arr.shape[0]
            xs = []
            ys = []
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    xs.append(w * elem)
                else:
                    ys.append(h * elem)

            assert len(xs) > 4, f"{save_dir} {image_name} {line} {parts}"
            centers.append(np.array([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2]))

    with open(os.path.join(save_dir, 'labels', image_name + '.txt'), 'r') as f:
        os.makedirs(os.path.join(save_dir, 'Crops', image_name), exist_ok=True)

        for line in f:
            loc_crop[idx] = {}
            crpfld = os.path.join(save_dir, 'Crops', image_name, str(idx))
            os.makedirs(crpfld, exist_ok=True)

            parts = line.split(' ')
            if len(parts) < 4:
                continue
            conf = float(parts[-1])
            cat = parts[0]
            poly = [float(x) for x in parts[1:-1]]

            loc_crop[idx]['cat'] = cat
            loc_crop[idx]['conf'] = conf
            loc_crop[idx]['fld'] = crpfld
            with open(os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(line)

            with open(os.path.join(crpfld, f'conf{str(idx).zfill(3)}.txt'), 'w') as f2:
                f2.write(str(conf))


            if show_indiv:
                viz_polygon(os.path.join(save_dir, 'input', resf), os.path.join(crpfld, f'poly{str(idx).zfill(3)}.txt'),
                            os.path.join(crpfld, f'poly{str(idx).zfill(3)}.png'))

            poly_resc = []
            w = arr.shape[1]
            h = arr.shape[0]
            xs = []
            ys = []
            for i, elem in enumerate(poly):
                if i % 2 == 0:
                    poly_resc.append(w * elem)
                    xs.append(w * elem)
                else:
                    poly_resc.append(h * elem)
                    ys.append(h * elem)

            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)



            polygon_resc = []
            for x, y in zip(xs, ys):
                polygon_resc.append((x, y))

            # angle, rect_w, rect_h, rect_iou = approximate_poly_as_rect(polygon_resc, savefld=crpfld, angle_res=20, iterations=1)
            # angle, rect_w, rect_h, rect_iou = approximate_poly_as_flat_rect(polygon_resc, savefld=crpfld, angle_res=20, iterations=2)
            angle, rect_w, rect_h, rect_iou = approximate_poly_as_flat_rect(polygon_resc, savefld=None, angle_res=20, iterations=2)

            loc_crop[idx]['rect_angle'] = angle
            loc_crop[idx]['rect_width_nm'] = rect_w * NM_P_PX
            loc_crop[idx]['rect_height_nm'] = rect_h * NM_P_PX
            loc_crop[idx]['rect_area_nm'] = rect_h * rect_w * NM_P_PX**2

            loc_crop[idx]['rect_iou'] = rect_iou

            polygon_resc = Polygon(polygon_resc)
            centroid_x = polygon_resc.centroid.x * NM_P_PX
            centroid_y = polygon_resc.centroid.y * NM_P_PX

            points = []

            maxd = -np.inf
            max_pair = (0, len(xs) - 1)
            for x, y in zip(xs, ys):
                points.append(np.array([x, y]))

            for i in range(len(points)):
                for j in range(i):
                    if np.linalg.norm(points[i] - points[j]) > maxd:
                        maxd = np.linalg.norm(points[i] - points[j])
                        max_pair = (i, j)

            dir = points[max_pair[1]] - points[max_pair[0]]
            dir /= np.linalg.norm(dir)
            min_orth = np.inf
            max_orth = - np.inf

            sp = points[max_pair[0]]
            for point in points:
                dist = float(np.cross((point - sp), dir))
                min_orth = min(min_orth, dist)
                max_orth = max(max_orth, dist)

            orth_dist = max_orth - min_orth

            theta = np.arcsin(dir[1]) * 180 / np.pi

            center = np.array([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2])
            loc_crop[idx]['center_x'] = center[0] * NM_P_PX
            loc_crop[idx]['center_y'] = center[1] * NM_P_PX

            dists = []
            for c in centers:
                d = np.linalg.norm(center - c) * NM_P_PX
                if d > 0:
                    dists.append(d)
            dists = sorted(dists)

            while (len(dists)) < 5:
                dists.append(max(w, h) * NM_P_PX)

            with open(os.path.join(crpfld, f'polyRESC{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{cat} {' '.join([str(x) for x in poly_resc])} {conf}\n")

            with open(os.path.join(crpfld, f'polyRESC{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{cat} {' '.join([str(x) for x in poly_resc])} {conf}\n")

            area = calc_poly_area(poly_resc)
            area_nm = area * NM_P_PX ** 2

            with open(os.path.join(crpfld, f'area{str(idx).zfill(3)}.txt'), 'w') as f3:
                f3.write(f"{area}px\n{area_nm} nm\n")

            loc_crop[idx]['area'] = area
            loc_crop[idx]['area_nm'] = area_nm
            loc_crop[idx]['w'] = w
            loc_crop[idx]['h'] = h
            loc_crop[idx]['centroid_x'] = centroid_x
            loc_crop[idx]['centroid_y'] = centroid_y
            loc_crop[idx]['nn1'] = dists[0]
            loc_crop[idx]['nn2'] = dists[1]
            loc_crop[idx]['nn3'] = dists[2]
            loc_crop[idx]['nn4'] = dists[3]
            loc_crop[idx]['nn5'] = dists[4]
            loc_crop[idx]['span_x_nm'] = span_x * NM_P_PX
            loc_crop[idx]['span_y_nm'] = span_y * NM_P_PX
            loc_crop[idx]['long_nm'] = maxd * NM_P_PX
            loc_crop[idx]['short_nm'] = orth_dist * NM_P_PX
            loc_crop[idx]['alignment'] = theta



            idx += 1

        # Eval On Grid_Percentage
        x_vals = []
        y_vals = []
        confs = []
        for k in loc_crop.keys():
            x_vals.append(loc_crop[k]['centroid_x'])
            y_vals.append(loc_crop[k]['centroid_y'])
            confs.append(loc_crop[k]['conf'])

        points = np.array([(x, y) for x, y in zip(x_vals, y_vals)])
        x_distances = []
        for x1 in x_vals:
            for x2 in x_vals:
                for p in [1]:
                    if x1 != x2:
                        x_distances.append(abs(x1 - x2))

        y_distances = []
        for x1 in y_vals:
            for x2 in y_vals:
                for p in [1]:
                    if x1 != x2:
                        y_distances.append(abs(x1 - x2))

        hx = np.histogram(x_distances, bins=100, density=True)
        hy = np.histogram(y_distances, bins=100, density=True)
        desn_x = hx[0]
        vals_x = [(hx[1][i] + hx[1][i + 1]) / 2 for i in range(len(desn_x))]
        desn_y = hy[0]
        vals_y = [(hy[1][i] + hy[1][i + 1]) / 2 for i in range(len(desn_y))]
        pairs_x = list(zip(vals_x, desn_x))
        pairs_y = list(zip(vals_y, desn_y))
        try:
            fx = filter_pairs(pairs_x)
            fy = filter_pairs(pairs_y)
            kpx = find_keypoint(x_vals, confs, fx)
            kpy = find_keypoint(y_vals, confs, fy)
        except Exception as e:
            print(e)
            fx = 0
            fy = 0
            kpx = 0
            kpy = 0


        no_grid =  fx < 0.05 * loc_crop[list(loc_crop.keys())[0]]['w'] * NM_P_PX or fy < 0.05 * loc_crop[list(loc_crop.keys())[0]]['h'] * NM_P_PX

        if not no_grid:

            grid_maxdist = np.sqrt((fx/2)**2 + (fy/2)**2)

            def gdx(x):
                dx = abs(x - kpx)
                per = round(dx / fx)
                clx = kpx + per * fx
                return abs(clx - x)

            def gdy(y):
                dy = abs(y - kpy)
                per = round(dy / fy)
                cly = kpy + per * fy
                return abs(cly - y)
            def grid_dist(x, y):
                dx = abs(x - kpx)
                per = round(dx / fx)
                clx = kpx + per * fx
                dy = abs(y - kpy)
                pery = round(dy / fy)
                cly = kpy + pery * fy

                pt = np.array([x, y])
                grid = np.array([clx, cly])
                return np.linalg.norm(pt - grid)


            for idx in loc_crop.keys():
                x = loc_crop[idx]['centroid_x']
                y = loc_crop[idx]['centroid_y']
                loc_crop[idx]['grid_dist_x_nm'] = gdx(x)
                loc_crop[idx]['grid_dist_y_nm'] = gdy(y)
                loc_crop[idx]['grid_dist_nm'] = grid_dist(x, y)
                loc_crop[idx]['grid_period_x_nm'] = fx
                loc_crop[idx]['grid_period_y_nm'] = fy
                loc_crop[idx]['on_grid_percentage'] = 1 - (grid_dist(x, y) / grid_maxdist)
                loc_crop[idx]['on_grid'] = 1 if (1 - (grid_dist(x, y) / grid_maxdist) > 0.9) else 0

        else:
            for idx in loc_crop.keys():
                loc_crop[idx]['grid_dist_x_nm'] = 0
                loc_crop[idx]['grid_dist_y_nm'] = 0
                loc_crop[idx]['grid_dist_nm'] = 0
                loc_crop[idx]['grid_period_x_nm'] = 0
                loc_crop[idx]['grid_period_y_nm'] = 0
                loc_crop[idx]['on_grid_percentage'] = 0
                loc_crop[idx]['on_grid'] = -1




    with open(out_json, 'w') as f:
        json.dump(loc_crop, f)

def apply_imageENTIRE_Set(imglist, weights, resultf=None, conf_thres=0.25, iou_thres=0.45, plain=True,img_idx = 0):
    save_dir = Path('Results') if resultf is None else resultf
    tempf = os.path.join(save_dir, 'Crops')
    os.makedirs(tempf, exist_ok=True)

    inp_arr = os.path.join(save_dir, 'input')
    os.makedirs(inp_arr, exist_ok=True)
    pairs = []

    imgsize = (3072, 2048)
    imgsize = (2048, 3072)

    img_namedict = {}

    inv_fld = os.path.join(save_dir, 'cropInv')
    os.makedirs(inv_fld, exist_ok=True)
    ps = []

    arglist = []
    for i, elem in enumerate(imglist):
        fnin = elem
        fnout = os.path.join(inv_fld, f"{str(i).zfill(4)}.png")
        img_namedict[os.path.basename(fnout)] = fnin
        if os.path.isfile(fnout):
            continue
        # ps.append(Process(target=cinv, args=(fnin, fnout)))
        # ps[-1].start()
        arglist.append((fnin, fnout))

    with Pool(THREADS) as p:
        for _ in tqdm(p.imap_unordered(cinv, arglist), total=len(arglist), desc='Crop + inv'):
            pass
    os.makedirs(os.path.join(save_dir, 'preds'), exist_ok=True)





    for elem in os.listdir(inv_fld):
        pairs.append((os.path.join(inv_fld, elem), os.path.join(inp_arr, elem)))


    PreprocessImages.pp_list_parallel(pairs)

    # inp2 = os.path.join(save_dir, 'input2')
    # os.makedirs(inp2, exist_ok=True)
    # for elem in pairs:
    #     fnin = elem[1]
    #     shutil.copy(fnin, os.path.join(inp2, os.path.basename(fnin)))

    if len(pairs) != len(os.listdir(os.path.join(save_dir, 'preds'))):
        visualize = False



        # plt.imsave(os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png'), arr, cmap='gray', vmin=0, vmax=1)
        # infile_file = os.path.join(save_dir, f'{str(img_idx).zfill(4)}_in.png')
        # shutil.copy(os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png'), infile_file)
        # imagefn = os.path.join(save_dir, f'{str(img_idx).zfill(4)}.png')

        # image_name = os.path.basename(imagefn).split('.')[0]



        source = inp_arr
        save_img = True  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        # Directories
        Path(os.path.join(save_dir,"labels")).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device()
        # print('DEVICE: ', device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsize, s=stride)  # check image size
        vid_stride = 1

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        view_img = True

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(os.path.join(save_dir, Path(path).stem), mkdir=True) if visualize else False
                pred, proto = model(im, augment=False, visualize=visualize)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path

                save_path = str(save_dir / "preds" / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                imc = im0  # for save_crop
                annotator = Annotator(im0, line_width=1, example=str(names))
                if len(det):

                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Segments
                    segments = [
                        scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))
                    ]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting
                    annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=im[i],
                    )

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

                # Stream results
                im0 = annotator.result()
                if False:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                        exit()

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    crops = {}

    ps = []
    rsfs = list(os.listdir(os.path.join(save_dir, 'preds')))
    out_jsons =os.path.join(save_dir, 'json_res')
    os.makedirs(out_jsons, exist_ok=True)
    old = 0

    arglist = []
    for resf in rsfs:
        arglist.append((resf, save_dir))


    # return
    # with Pool(THREADS) as p:
    #     for _ in tqdm(p.imap_unordered(analyze_result, arglist), total=len(arglist), desc='Eval Results'):
    #         pass

    for al in tqdm(arglist, desc='Eval Non parallel'):
        visualize_preds(al)



    for resf in tqdm(list(os.listdir(os.path.join(save_dir, 'preds'))), desc='Load Results'):
        originalName = img_namedict[resf]
        img_name = resf.split('.')[0]
        with open(os.path.join(out_jsons, img_name + '.json'), 'rb') as f:
            d = json.load(f)
        crops[originalName] = d

    return crops
# def analye_folders(fld, resultf, weights, conf_thres=0.25, iou_thres=0.45, plain=True):
#     files = []
#     flds = []
#
#     os.makedirs(resultf, exist_ok=True)
#
#     for elem in os.listdir(fld):
#         f = os.path.join(fld, elem)
#         if os.path.isfile(f):
#             files.append(f)
#         else:
#             flds.append(f)
#
#     while len(flds) > 0:
#         floc = flds.pop()
#         for elem in os.listdir(floc):
#             f = os.path.join(floc, elem)
#             if os.path.isfile(f):
#                 files.append(f)
#             else:
#                 flds.append(f)
#
#
#     # files = files[::50]
#     image_res = {}
#     for i, fn in tqdm(enumerate(files), total=len(files)):
#         image_res[i] = {}
#         image_res[i]['fn'] = fn
#         base = os.path.basename(fn)
#         if 'x' in base:
#             parts = base.split('x', maxsplit=1)
#             p1 = parts[0]
#             if ' ' in parts[1]:
#                 p2 = parts[1].split(' ')[0]
#             elif '_' in parts[1]:
#                 p2 = parts[1].split('_')[0]
#             else:
#                 p2 = parts[1].split('.')[0]
#             image_res[i]['p1'] = p1
#             image_res[i]['p2'] = p2
#         else:
#             image_res[i]['p1'] = 0
#             image_res[i]['p2'] = 0
#
#
#
#         results = apply_imageENTIRE(fn, weights, resultf=resultf, conf_thres=conf_thres, iou_thres=iou_thres, plain=plain, img_idx = i)
#         image_res[i]['res'] = results
#
#     with open(os.path.join(resultf, 'res.json'), "w") as outfile:
#         json.dump(image_res, outfile)
#
#     json_2_csv(os.path.join(resultf, 'res.json'), os.path.join(resultf, 'res.csv'))


def analye_folders_Set(fld, resultf, weights, conf_thres=0.25, iou_thres=0.45, plain=True):
    files = []
    flds = []



    os.makedirs(resultf, exist_ok=True)

    for elem in os.listdir(fld):
        f = os.path.join(fld, elem)
        if os.path.isfile(f):
            files.append(f)
        else:
            flds.append(f)

    while len(flds) > 0:
        floc = flds.pop()
        for elem in os.listdir(floc):
            f = os.path.join(floc, elem)
            if os.path.isfile(f):
                files.append(f)
            else:
                flds.append(f)

    # files = files[::100]
    results_total = apply_imageENTIRE_Set(files, weights, resultf=resultf, conf_thres=conf_thres, iou_thres=iou_thres,
                                    plain=plain)


    image_res = {}
    for i, fn in tqdm(enumerate(files), total=len(files)):
        image_res[i] = {}
        image_res[i]['fn'] = fn
        base = os.path.basename(fn)
        if 'x' in base:
            parts = base.split('x', maxsplit=1)
            p1 = parts[0]
            if ' ' in parts[1]:
                p2 = parts[1].split(' ')[0]
            elif '_' in parts[1]:
                p2 = parts[1].split('_')[0]
            else:
                p2 = parts[1].split('.')[0]
            image_res[i]['p1'] = p1
            image_res[i]['p2'] = p2
        else:
            image_res[i]['p1'] = 0
            image_res[i]['p2'] = 0

            # Dims Folder
        fldname = os.path.basename(os.path.dirname(fn))
        if 'x' in fldname:
            parts = fldname.split('x', maxsplit=1)
            p1 = parts[0]
            if ' ' in parts[1]:
                p2 = parts[1].split(' ')[0]
            elif '_' in parts[1]:
                p2 = parts[1].split('_')[0]
            elif '.' in parts[1]:
                p2 = parts[1].split('.')[0]
            else:
                p2 = parts[1].strip()
            image_res[i]['p1_fld'] = p1
            image_res[i]['p2_fld'] = p2
        else:
            image_res[i]['p1_fld'] = 0
            image_res[i]['p2_fld'] = 0


        if 'short' in fn:
            image_res[i]['short_axis'] = 1
            image_res[i]['long_axis'] = 0
        elif 'long' in fn:
            image_res[i]['short_axis'] = 0
            image_res[i]['long_axis'] = 1
        else:
            image_res[i]['short_axis'] = 0
            image_res[i]['long_axis'] = 0

        if image_res[i]['short_axis'] + image_res[i]['long_axis'] == 1:
            image_res[i]['p_x'] = image_res[i]['p1'] if image_res[i]['short_axis'] == 1 else image_res[i]['p2']
            image_res[i]['p_y'] = image_res[i]['p2'] if image_res[i]['short_axis'] == 1 else image_res[i]['p1']
        else:
            image_res[i]['p_x'] = 0
            image_res[i]['p_y'] = 0


        results = results_total[fn]

        image_res[i]['res'] = results

    with open(os.path.join(resultf, 'res.json'), "w") as outfile:
        json.dump(image_res, outfile)

    json_2_csv(os.path.join(resultf, 'res.json'), os.path.join(resultf, 'res.csv'))


def json_2_csv(jsonfile, csv_res):
    with open(jsonfile, 'rb') as f:
        d = json.load(f)
    df = {}

    tid = 0
    total_ids = []
    image_ids = []
    crop_ids = []
    image_fns = []
    image_p1 = []
    image_p2 = []
    image_p1_fld = []
    image_p2_fld = []
    image_short = []
    image_long = []
    p_xs = []
    p_ys = []
    cats = []
    confs = []
    flds = []
    ws = []
    hs = []
    areas = []
    area_nms = []
    nn1s = []
    nn2s = []
    nn3s = []
    nn4s = []
    nn5s = []
    cx = []
    cy = []
    spx = []
    spy = []
    wdl = []
    wds = []
    thetas = []
    centroid_x = []
    centroid_y = []
    gdx = []
    gdy = []
    gpx = []
    gpy = []
    dgp = []
    og = []
    rect_angle = []
    rect_w = []
    rect_h = []
    rect_areas = []
    rect_iou = []



    for img_id in tqdm(d.keys()):
        for cid in d[img_id]['res'].keys():
            total_ids.append(tid)
            tid += 1
            image_ids.append(img_id)
            crop_ids.append(cid)
            image_fns.append(d[img_id]['fn'])
            image_p1.append(d[img_id]['p1'])
            image_p2.append(d[img_id]['p2'])
            image_p1_fld.append(d[img_id]['p1_fld'])
            image_p2_fld.append(d[img_id]['p2_fld'])
            image_short.append(d[img_id]['short_axis'])
            image_long.append(d[img_id]['long_axis'])
            p_xs.append(d[img_id]['p_x'])
            p_ys.append(d[img_id]['p_y'])
            cats.append(d[img_id]['res'][cid]['cat'])
            confs.append(d[img_id]['res'][cid]['conf'])
            flds.append(d[img_id]['res'][cid]['fld'])
            ws.append(d[img_id]['res'][cid]['w'])
            hs.append(d[img_id]['res'][cid]['h'])
            nn1s.append(d[img_id]['res'][cid]['nn1'])
            nn2s.append(d[img_id]['res'][cid]['nn2'])
            nn3s.append(d[img_id]['res'][cid]['nn3'])
            nn4s.append(d[img_id]['res'][cid]['nn4'])
            nn5s.append(d[img_id]['res'][cid]['nn5'])
            centroid_x.append(d[img_id]['res'][cid]['centroid_x'])
            centroid_y.append(d[img_id]['res'][cid]['centroid_y'])
            cx.append(d[img_id]['res'][cid]['center_x'])
            cy.append(d[img_id]['res'][cid]['center_y'])
            areas.append(d[img_id]['res'][cid]['area'])
            area_nms.append(d[img_id]['res'][cid]['area_nm'])
            spx.append(d[img_id]['res'][cid]['span_x_nm'])
            spy.append(d[img_id]['res'][cid]['span_y_nm'])
            wdl.append(d[img_id]['res'][cid]['long_nm'])
            wds.append(d[img_id]['res'][cid]['short_nm'])
            thetas.append(d[img_id]['res'][cid]['alignment'])
            gdx.append(d[img_id]['res'][cid]['grid_dist_x_nm'])
            gdy.append(d[img_id]['res'][cid]['grid_dist_y_nm'])
            gpx.append(d[img_id]['res'][cid]['grid_period_x_nm'])
            gpy.append(d[img_id]['res'][cid]['grid_period_y_nm'])
            dgp.append(d[img_id]['res'][cid]['on_grid_percentage'])
            og.append(d[img_id]['res'][cid]['on_grid'])
            rect_angle.append(d[img_id]['res'][cid]['rect_angle'])
            rect_w.append(d[img_id]['res'][cid]['rect_width_nm'])
            rect_h.append(d[img_id]['res'][cid]['rect_height_nm'])
            rect_iou.append(d[img_id]['res'][cid]['rect_iou'])
            rect_areas.append(d[img_id]['res'][cid]['rect_area_nm'])


            #loc_crop[idx]['grid_dist_x_nm'] = 0
            #     loc_crop[idx]['grid_dist_y_nm'] = 0
            #     loc_crop[idx]['grid_dist_nm'] = 0
            #     loc_crop[idx]['grid_period_x_nm'] = 0
            #     loc_crop[idx]['grid_period_y_nm'] = 0
            #     loc_crop[idx]['on_grid_percentage'] = 0
            #     loc_crop[idx]['on_grid'] = None



    df['total_id'] = total_ids
    df['img_id'] = image_ids
    df['crop_id'] = crop_ids
    df['img_fn'] = image_fns
    df['crop_fn'] = flds
    df['p1'] = image_p1
    df['p2'] = image_p2
    df['p1_folder'] = image_p1_fld
    df['p2_folder'] = image_p2_fld
    df['short_axis'] = image_short
    df['long_axis'] = image_long
    df['p_xdir'] = p_xs
    df['p_ydir'] = p_ys
    df['cat'] = cats
    df['conf'] = confs
    df['w_px'] = ws
    df['h_px'] = hs
    df['grid_dist_x_nm'] = gdx
    df['grid_dist_y_nm'] = gdy
    df['grid_period_x_nm'] = gpx
    df['grid_period_y_nm'] = gpy
    df['on_grid_percentage'] = dgp
    df['on_grid'] = og
    df['mid_x_nm'] = cx
    df['mid_y_nm'] = cy
    df['centroid_x_nm'] = centroid_x
    df['centroid_y_nm'] = centroid_y
    df['NN1_nm'] = nn1s
    df['NN2_nm'] = nn2s
    df['NN3_nm'] = nn3s
    df['NN4_nm'] = nn4s
    df['NN5_nm'] = nn5s
    df['area'] = areas
    df['area_nm'] = area_nms
    df['span_x_nm'] = spx
    df['span_y_nm'] = spy
    df['span_long_axis_nm'] = wdl
    df['span_short_axis_nm'] = wds
    df['long_axis_angle'] = thetas
    df['rect_angle'] = rect_angle
    df['rect_width_nm'] = rect_w
    df['rect_height_nm'] = rect_h
    df['rect_area_nm'] = rect_areas
    df['rect_iou'] = rect_iou



    df = pd.DataFrame(df)
    df.to_csv(csv_res, sep=';')

    write_cat_csvs(csv_res, os.path.join(resultf, 'csvs'), conf_th=0.8)


def write_cat_csvs(file, resfld, conf_th=0.8):
    prime1 = 389
    prime2 = 13003
    os.makedirs(resfld, exist_ok=True)
    dicts = {}
    df = pd.read_csv(file, sep=';')
    for i in range(len(df['total_id'])):
        if df['conf'][i] < conf_th:
            continue
        magic = df['p1'][i] * prime1 + df['p2'][i] * prime2
        if magic not in dicts.keys():
            dicts[magic] = {'file': [],
                            'p1': [],
                            'p2': [],
                            'conf': [],
                            'span_x_nm': [],
                            'span_y_nm': [],
                            'rect_x_nm': [],
                            'rect_y_nm': []}
        dicts[magic]['file'].append(df['img_fn'][i])
        dicts[magic]['p1'].append(df['p1'][i])
        dicts[magic]['p2'].append(df['p2'][i])
        dicts[magic]['conf'].append(df['conf'][i])
        dicts[magic]['span_x_nm'].append(df['span_x_nm'][i])
        dicts[magic]['span_y_nm'].append(df['span_y_nm'][i])
        dicts[magic]['rect_x_nm'].append(df['rect_width_nm'][i])
        dicts[magic]['rect_y_nm'].append(df['rect_height_nm'][i])

    for k in dicts.keys():
        fn = os.path.join(resfld, f'{dicts[k]["p1"][0]}x{dicts[k]["p2"][0]}.csv')
        pd.DataFrame(dicts[k]).to_csv(fn, sep=';')


if __name__ == "__main__":
    # imagef = r"D:\seifert\PycharmProjects\LiuDimer\Data\Data library\Dimer in long axis\70x50\70x50 (6).tif"
    weights = r'D:\seifert\PycharmProjects\LiuDimer\yolov5-master\yolov5-master\runs\train-seg\exp13\weights\last.pt'
    weights = r'G:\seife\PycharmG\LiuDimerAI\yolov5-master\yolov5-master\runs\train-seg\exp13\weights\last.pt'
    # resultf = Path(r'C:\Users\seifert\Pictures\Temp\Out\Sizes4')
    # resultf = Path(r"F:\Data\LiuDimer\VisualizePred_NonInv")
    resultf = Path(r"F:\Data\Musfira_200226\Result")

    os.makedirs(resultf, exist_ok=True)

    # datafld = r"D:\seifert\PycharmProjects\LiuDimer\Data\Data library"
    # datafld = r'C:\Users\seifert\Pictures\Temp\DataLib\comparison'
    datafld = r"F:\Data\Musfira_200226\Data"
    # datafld = r'D:\seifert\PycharmProjects\LiuDimer\SynthData\Test108'

    # apply_imageENTIRE(imagef, weights)

    # json_2_csv(r"D:\seifert\PycharmProjects\LiuDimer\yolov5-master\yolov5-master\segment\Results\res.json", r"D:\seifert\PycharmProjects\LiuDimer\yolov5-master\yolov5-master\segment\Results\res.csv")

    # assert 4 == 5

    analye_folders_Set(fld=datafld,
                   resultf=resultf,
                   weights=weights,
                   conf_thres=0.7,
                   iou_thres=0.45,
                   plain=False)


    #opt = parse_opt()
    #main(opt)
