import argparse
import cv2
import os
import logging
from pathlib import Path
import sys
import numpy as np
import psutil
import time
import math
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, TclError

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, xywh2xyxy, clip_coords, increment_path, strip_optimizer,
                                  colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        f = str(Path(increment_path(file)).with_suffix('.jpg'))
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(f, quality=95, subsampling=0)
    return crop

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

@torch.no_grad()
def run(
        source='/home/splab/Downloads/projintern/myproject/lab30s.mp4',
        yolo_weights=WEIGHTS / 'yolov7x.pt',
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),
        conf_thres=0.3,
        iou_thres=0.4,
        max_det=1000,
        device='',
        show_vid=True,
        save_txt=True,
        save_conf=False,
        save_crop=True,
        save_vid=True,
        nosave=False,
        classes=[0,39,62,63,64,65,66,67],
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'outputs_intern',
        name='tracking_results',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        hide_class=False,
        half=True,
        dnn=False,
        frame_skip=1
):
    source = str(source)

    # Fix Path conversion issue
    if isinstance(yolo_weights, str):
        yolo_weights = Path(yolo_weights)

    if isinstance(strong_sort_weights, str):
        strong_sort_weights = Path(strong_sort_weights)

    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    if not isinstance(yolo_weights, list):
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize Tkinter GUI
    root = tk.Tk()
    root.title("Tracking Dashboard")
    root.geometry("1280x736")
    root.configure(bg="#333333")

    # Video panel (left)
    video_frame = ttk.Frame(root, style="Video.TFrame")
    video_frame.pack(side="left", padx=10, pady=10)
    ttk.Label(video_frame, text="Video Feed", style="Title.TLabel").pack(pady=5)
    video_label = ttk.Label(video_frame)
    video_label.pack()

    # Features panel (right)
    features_frame = ttk.Frame(root, relief="raised", borderwidth=2, style="Features.TFrame")
    features_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # Style configuration
    style = ttk.Style()
    style.configure("Video.TFrame", background="#333333")
    style.configure("Features.TFrame", background="#333333")
    style.configure("Title.TLabel", font=("Arial", 14, "bold"), foreground="#ffffff", background="#333333")
    style.configure("FPS.TLabel", font=("Arial", 12), foreground="#00cc00", background="#333333")
    style.configure("Person.TLabel", font=("Arial", 12), foreground="#ff6666", background="#333333")
    style.configure("Info.TLabel", font=("Arial", 12), foreground="#cccccc", background="#333333")
    style.configure("Track.TLabel", font=("Arial", 10), foreground="#66ccff", background="#333333")
    style.configure("Count.TLabel", font=("Arial", 12), foreground="#ff6666", background="#333333")
    style.configure("Total.TLabel", font=("Arial", 12, "bold"), foreground="#00cc00", background="#333333")
    style.configure("Button.TButton", font=("Arial", 10), background="#3B4D4D", foreground="#39394E", width=12, padding=3)

    # Feature labels
    ttk.Label(features_frame, text="Tracking Features", style="Title.TLabel").pack(pady=5)
    fps_label = ttk.Label(features_frame, text="FPS: 0.0 | Memory: 0.0 MB", style="FPS.TLabel")
    fps_label.pack(anchor="w", padx=10, pady=2)
    person_label = ttk.Label(features_frame, text="Persons: 0", style="Person.TLabel")
    person_label.pack(anchor="w", padx=10, pady=2)
    frame_label = ttk.Label(features_frame, text="Frame: 0", style="Info.TLabel")
    frame_label.pack(anchor="w", padx=10, pady=2)
    yolo_time_label = ttk.Label(features_frame, text="YOLO Time: 0.000s", style="Info.TLabel")
    yolo_time_label.pack(anchor="w", padx=10, pady=2)
    sort_time_label = ttk.Label(features_frame, text="StrongSORT Time: 0.000s", style="Info.TLabel")
    sort_time_label.pack(anchor="w", padx=10, pady=2)
    entered_label = ttk.Label(features_frame, text="Entered: 0", style="Count.TLabel")
    entered_label.pack(anchor="w", padx=10, pady=2)
    left_label = ttk.Label(features_frame, text="Left: 0", style="Count.TLabel")
    left_label.pack(anchor="w", padx=10, pady=2)
    total_label = ttk.Label(features_frame, text="Total in Room: 0", style="Total.TLabel")
    total_label.pack(anchor="w", padx=10, pady=2)

    # Doorway line controls
    controls_frame = ttk.Frame(features_frame, style="Features.TFrame")
    controls_frame.pack(anchor="w", padx=10)

    # Line position controls
    line_pos_frame = ttk.Frame(controls_frame, style="Features.TFrame")
    line_pos_frame.pack(side="left", padx=5)
    ttk.Label(line_pos_frame, text="Line Position", style="Info.TLabel").pack(anchor="w")
    doorway_line_ratio = 0.6
    line_position_label = ttk.Label(line_pos_frame, text=f"Line Position: {doorway_line_ratio*100:.0f}%", style="Info.TLabel")
    line_position_label.pack(anchor="w")
    def move_line_up():
        nonlocal doorway_line_ratio
        doorway_line_ratio = max(0.0, doorway_line_ratio - 0.05)
        line_position_label.configure(text=f"Line Position: {doorway_line_ratio*100:.0f}%")
    def move_line_down():
        nonlocal doorway_line_ratio
        doorway_line_ratio = min(1.0, doorway_line_ratio + 0.05)
        line_position_label.configure(text=f"Line Position: {doorway_line_ratio*100:.0f}%")
    ttk.Button(line_pos_frame, text="Up", command=move_line_up, style="Button.TButton").pack(anchor="w", padx=2, pady=2)
    ttk.Button(line_pos_frame, text="Down", command=move_line_down, style="Button.TButton").pack(anchor="w", padx=2, pady=2)

    # Slope controls
    slope_frame = ttk.Frame(controls_frame, style="Features.TFrame")
    slope_frame.pack(side="left", padx=5)
    ttk.Label(slope_frame, text="Slope Angle", style="Info.TLabel").pack(anchor="w")
    doorway_line_angle = 0.0
    slope_label = ttk.Label(slope_frame, text=f"Slope Angle: {doorway_line_angle:.0f}°", style="Info.TLabel")
    slope_label.pack(anchor="w")
    def increase_slope():
        nonlocal doorway_line_angle
        doorway_line_angle = min(45.0, doorway_line_angle + 2.5)
        slope_label.configure(text=f"Slope Angle: {doorway_line_angle:.0f}°")
    def decrease_slope():
        nonlocal doorway_line_angle
        doorway_line_angle = max(-45.0, doorway_line_angle - 2.5)
        slope_label.configure(text=f"Slope Angle: {doorway_line_angle:.0f}°")
    ttk.Button(slope_frame, text="Increase", command=increase_slope, style="Button.TButton").pack(anchor="w", padx=2, pady=2)
    ttk.Button(slope_frame, text="Decrease", command=decrease_slope, style="Button.TButton").pack(anchor="w", padx=2, pady=2)

    # Scrollable tracks
    tracks_frame = ttk.Frame(features_frame, style="Features.TFrame")
    tracks_frame.pack(fill="both", expand=True, padx=10, pady=5)
    tracks_canvas = tk.Canvas(tracks_frame, bg="#333333")
    tracks_scrollbar = ttk.Scrollbar(tracks_frame, orient="vertical", command=tracks_canvas.yview)
    tracks_scrollable_frame = ttk.Frame(tracks_canvas)
    tracks_scrollable_frame.bind(
        "<Configure>",
        lambda e: tracks_canvas.configure(scrollregion=tracks_canvas.bbox("all"))
    )
    tracks_canvas.create_window((0, 0), window=tracks_scrollable_frame, anchor="nw")
    tracks_canvas.configure(yscrollcommand=tracks_scrollbar.set)
    tracks_canvas.pack(side="left", fill="both", expand=True)
    tracks_scrollbar.pack(side="right", fill="y")
    track_labels = []

    device = select_device(device)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(yolo_weights, map_location=device)
    names = model.names
    stride = model.stride.max().cpu().numpy()
    imgsz = check_img_size(imgsz[0], s=stride)

    if half and device.type != 'cpu':
        model.half()
    else:
        half = False
        LOGGER.info('Half-precision disabled (CPU or incompatible GPU)')

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset.sources)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer = [None] * nr_sources, [None] * nr_sources

    cfg = get_config()
    cfg.merge_from_file(config_strongsort)
    cfg.STRONGSORT.MAX_AGE = 15
    cfg.STRONGSORT.N_INIT = 2

    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources

    colors={0:(0,255,0),39:(255,0,0),62:(255,255,0),63:(0,0,255),64:(255,0,255),65:(0,255,255),66:(128,255,0),67:(255,128,0)}

    # People counting variables
    entered_count = 0
    left_count = 0
    id_states = {}
    ID_TIMEOUT = 5.0

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    t1 = time.time()
    try:
        for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
            if frame_idx % frame_skip != 0:
                continue
            if im is None or im0s is None:
                LOGGER.error(f"Frame {frame_idx}: im0 is None or has no shape, skipping people counting")
                continue

            s = ''
            t2 = time_synchronized()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            t3 = time_synchronized()
            dt[0] += t3 - t2

            visualize = str(increment_path(save_dir / Path(path[0]).stem, mkdir=True)) if visualize else False
            pred = model(im)
            t4 = time_synchronized()
            dt[1] += t4 - t3

            pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
            dt[2] += time_synchronized() - t4

            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name) + str(i)
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)
                    else:
                        txt_file_name = p.parent.name
                        save_path = str(save_dir / p.parent.name)

                curr_frames[i] = im0
                txt_path = str(save_dir / 'tracks' / txt_file_name)
                s += '%gx%g ' % im.shape[2:]
                imc = im0.copy() if save_crop else im0

                if cfg.STRONGSORT.ECC:
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                tracks = []
                person_count = 0
                t5 = time_synchronized()
                t6 = t5
                if det is not None and len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        if int(c) == 0:
                            person_count = n.item()

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t6 = time_synchronized()
                    dt[3] += t6 - t5

                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            x1, y1, x2, y2 = bboxes
                            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                            track_obj = strongsort_list[i].tracker.tracks[j]
                            track_age = track_obj.age
                            track_state = "Confirmed" if track_obj.is_confirmed() else "Tentative"
                            tracks.append({
                                "id": int(id),
                                "bbox": [float(x), float(y), float(w), float(h)],
                                "class": int(cls),
                                "confidence": float(conf),
                                "age": int(track_age),
                                "state": track_state
                            })

                            if save_txt:
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if save_vid or save_crop or show_vid:
                                c = int(cls)
                                id = int(id)
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else
                                                                 (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                plot_one_box(bboxes, im0, label=label, color=colors.get(c,(255,255,255)), line_thickness=2)
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                else:
                    strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')

                # People counting logic
                if im0 is None or not hasattr(im0, 'shape'):
                    LOGGER.error(f"Frame {frame_idx}: im0 is None or has no shape, skipping people counting")
                    continue
                m = math.tan(math.radians(doorway_line_angle))
                b = (doorway_line_ratio * im0.shape[0]) - m * (im0.shape[1] / 2)
                current_time = time.time()
                current_ids = set()
                total_in_room = 0
                for track in tracks:

                    # Count only PERSON class (COCO class 0)
                    if track['class'] != 0:
                        continue

                    track_id = track['id']
                    x, y, w, h = track['bbox']
                    y2 = y + h / 2
                    current_ids.add(track_id)
                    y_line = m * x + b
                    is_inside = y2 > y_line
                    if is_inside:
                        total_in_room += 1

                    if track_id not in id_states:
                        for tid, state in list(id_states.items()):
                            if tid == track_id and not state['active'] and (current_time - state['last_seen']) < ID_TIMEOUT:
                                id_states[track_id] = {
                                    'y2_prev': y2,
                                    'x_prev': x,
                                    'is_inside': is_inside,
                                    'counted': state['counted'],
                                    'last_seen': current_time,
                                    'active': True
                                }
                                LOGGER.info(f"Frame {frame_idx}: ID {track_id} restored, is_inside={is_inside}")
                                break
                        else:
                            id_states[track_id] = {
                                'y2_prev': y2,
                                'x_prev': x,
                                'is_inside': is_inside,
                                'counted': None,
                                'last_seen': current_time,
                                'active': True
                            }
                            LOGGER.info(f"Frame {frame_idx}: ID {track_id} initialized, x={x:.0f}, y2={y2:.0f}, y_line={y_line:.0f}, is_inside={is_inside}")
                    else:
                        y2_prev = id_states[track_id]['y2_prev']
                        x_prev = id_states[track_id]['x_prev']
                        is_inside_prev = id_states[track_id]['is_inside']
                        counted = id_states[track_id]['counted']
                        id_states[track_id]['last_seen'] = current_time
                        id_states[track_id]['active'] = True

                        y_line_prev = m * x_prev + b
                        if abs(y2 - y2_prev) > 0.5:
                            if y2_prev < y_line_prev and y2 > y_line and not is_inside_prev:
                                if counted == 'left':
                                    left_count = max(0, left_count - 1)
                                    LOGGER.info(f"Frame {frame_idx}: ID {track_id} re-entered, decremented left_count={left_count}")
                                entered_count += 1
                                id_states[track_id]['counted'] = 'entered'
                                LOGGER.info(f"Frame {frame_idx}: ID {track_id} entered, x={x:.0f}, y2={y2:.0f}, y_line={y_line:.0f}")
                            elif y2_prev > y_line_prev and y2 < y_line and is_inside_prev:
                                if counted == 'entered':
                                    entered_count = max(0, entered_count - 1)
                                    LOGGER.info(f"Frame {frame_idx}: ID {track_id} re-exited, decremented entered_count={entered_count}")
                                left_count += 1
                                id_states[track_id]['counted'] = 'left'
                                LOGGER.info(f"Frame {frame_idx}: ID {track_id} left, x={x:.0f}, y2={y2:.0f}, y_line={y_line:.0f}")

                        id_states[track_id]['is_inside'] = is_inside
                        id_states[track_id]['y2_prev'] = y2
                        id_states[track_id]['x_prev'] = x
                        LOGGER.info(f"Frame {frame_idx}: ID {track_id}, x={x:.0f}, y2={y2:.0f}, y_line={y_line:.0f}, is_inside={is_inside}, counted={counted}")

                expired_ids = []
                for track_id, state in list(id_states.items()):
                    if track_id not in current_ids:
                        state['active'] = False
                        state['last_seen'] = current_time
                    if not state['active'] and (current_time - state['last_seen']) > ID_TIMEOUT:
                        LOGGER.info(f"Frame {frame_idx}: ID {track_id} expired")
                        expired_ids.append(track_id)
                for track_id in expired_ids:
                    del id_states[track_id]

                fps = seen / (time.time() - t1) if seen > 0 else 0
                memory_info = psutil.Process().memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)
                fps_label.configure(text=f"FPS: {fps:.2f} | Memory: {memory_usage:.2f} MB")
                person_label.configure(text=f"Persons: {person_count}")
                frame_label.configure(text=f"Frame: {frame_idx + 1}")
                yolo_time_label.configure(text=f"YOLO Time: {(t4 - t3):.3f}s")
                sort_time_label.configure(text=f"StrongSORT Time: {(t6 - t5):.3f}s")
                entered_label.configure(text=f"Entered: {entered_count}")
                left_label.configure(text=f"Left: {left_count}")
                total_label.configure(text=f"Total in Room: {total_in_room}")

                for label in track_labels:
                    label.destroy()
                track_labels.clear()
                for idx, track in enumerate(tracks[:5]):
                    track_text = f"ID {track['id']}: [{track['bbox'][0]:.0f}, {track['bbox'][1]:.0f}, {track['bbox'][2]:.0f}, {track['bbox'][3]:.0f}], Conf: {track['confidence']:.2f}, Age: {track['age']}f, State: {track['state']}"
                    label = ttk.Label(tracks_scrollable_frame, text=track_text, style="Track.TLabel")
                    label.pack(anchor="w", pady=2)
                    track_labels.append(label)

                if show_vid:
                    im0_resized = cv2.resize(im0, (736, 414))
                    y1 = m * 0 + b
                    y2 = m * im0.shape[1] + b
                    y1_scaled = int(y1 * 414 / im0.shape[0])
                    y2_scaled = int(y2 * 414 / im0.shape[0])
                    cv2.line(im0_resized, (0, y1_scaled), (736, y2_scaled), (255, 0, 0), 2)
                    im0_rgb = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(im0_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)

                try:
                    root.update()
                except TclError:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if save_vid:

                    save_path = str(save_dir / "tracked_output.mp4")

                    if vid_writer[i] is None:

                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps = 30
                            h, w = im0.shape[:2]

                        print("\nSaving output to:")
                        print(save_path)

                        vid_writer[i] = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h)
                        )

                    vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]

            LOGGER.info(f'{s}Done. YOLO:({t4 - t3:.3f}s), StrongSORT:({t6 - t5:.3f}s)')

    finally:
        t = tuple(x / seen * 1E3 for x in dt) if seen > 0 else (0.0, 0.0, 0.0, 0.0)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)
        for vw in vid_writer:
            if vw is not None:
                vw.release()
        cv2.destroyAllWindows()
        try:
            root.destroy()
        except TclError:
            pass

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7x.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=str(WEIGHTS / 'osnet_x0_25_msmt17.pt'))
    parser.add_argument('--config-strongsort', type=str, default=str(ROOT / 'strong_sort/configs/strong_sort.yaml'))
    parser.add_argument('--source', type=str, default='/home/splab/Downloads/projintern/myproject/lab30s.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', default=True, help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', default=True, help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0,39,62,63,64,65,66,67], help='person,bottle,tv,laptop,mouse,remote,keyboard,mobile')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=str(ROOT / 'outputs_intern'), help='save results to project/name')
    parser.add_argument('--name', default='tracking_results', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', default=True, help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--frame-skip', type=int, default=1, help='process every nth frame for speed')
    opt = parser.parse_args()
    opt.imgsz = tuple(opt.imgsz) if len(opt.imgsz) == 2 else (opt.imgsz[0], opt.imgsz[0])
    return opt

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
