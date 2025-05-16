# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import os
import glob
import argparse
import cv2
import numpy as np
import ffmpeg
from functools import partial
from pathlib import Path
import json

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def Create_Videowriter(w, h, output_path):
    video_writer = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=12, s='{}x{}'.format(w, h))
        .output(output_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return video_writer

def Delete_Videowriter(video_writer):
    video_writer.stdin.close()

@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt'
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    yolo.predictor.custom_args = args
    
    cam_homographies, area_info = load_camera_info(args.camera_info_dir)
    
    save_dir = os.path.join(args.project,
                            os.path.basename(os.path.dirname(args.source)))
    # save_vid_path = os.path.join(args.project, 
    #                              os.path.basename(os.path.dirname(os.path.dirname(args.source))), 
    #                              os.path.basename(os.path.dirname(args.source)) + '.mp4')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for frame_id, r in enumerate(results):
        # if frame_id == 0:
        #     img = r.plot()
            # h, w = img.shape[:2]
            # video_writer = Create_Videowriter(w, h, save_vid_path)
        # video_writer.stdin.write(r.plot()[:, :, ::-1].tobytes())
        
        orig_img = r.orig_img
        
        if r.boxes and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            
            
            for id, box in zip(ids, boxes):
                id = int(id)
                x1, y1, x2, y2 = box.astype(int)
                
                center = calc_center(x1, y1, x2, y2, cam_homographies)
                area = which_area(center, area_info)
                
                if area in [8, 9, 10, 18]:
                    # save_txt_path = os.path.join(save_dir, "track.txt")
                    if not os.path.isdir(os.path.join(save_dir, "area" + str(area), "person" + str(id), "crops")):
                        os.makedirs(os.path.join(save_dir, "area" + str(area), "person" + str(id), "crops"), exist_ok=True)
                    if area == 8 and (x1 + x2)/2 >= 1280 / 2:
                        save_txt_path = os.path.join(save_dir, "area" + str(area), "person" + str(id), "track.txt")
                        if not os.path.isfile(save_txt_path):
                            with open(save_txt_path, 'w') as f:
                                f.write(f'{frame_id+1} {id} {area} {x1} {y1} {x2-x1} {y2-y1}\n')
                        else:
                            with open(save_txt_path, 'a') as f:
                                f.write(f'{frame_id+1} {id} {area} {x1} {y1} {x2-x1} {y2-y1}\n')
                                
                        image_crop = orig_img[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(save_dir, "area" + str(area), "person" + str(id), "crops", "{}.jpg".format(frame_id)), image_crop)
                    elif area == 10 and (x1 + x2)/2 < 1280 / 2:
                        save_txt_path = os.path.join(save_dir, "area" + str(area), "person" + str(id), "track.txt")
                        if not os.path.isfile(save_txt_path):
                            with open(save_txt_path, 'w') as f:
                                f.write(f'{frame_id+1} {id} {area} {x1} {y1} {x2-x1} {y2-y1}\n')
                        else:
                            with open(save_txt_path, 'a') as f:
                                f.write(f'{frame_id+1} {id} {area} {x1} {y1} {x2-x1} {y2-y1}\n')
                                
                        image_crop = orig_img[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(save_dir, "area" + str(area), "person" + str(id), "crops", "{}.jpg".format(frame_id)), image_crop)
                        
                
        # img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        # if hasattr(yolo.predictor.trackers[0], 'plot_results'):
        #     img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        # else:
        #     print("Tracker does not support plot_results, skipping visualization.")
        #     img = r.orig_img  # Pass the original image without annotations
        


        if args.show is True:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
            
    # Delete_Videowriter(video_writer)

def load_camera_info(info_dir):
    cam_info_path = glob.glob(os.path.join(info_dir, "CameraInfo", "*.json"))
    cam_homographies = {}
    for path in cam_info_path:
        cam_id = os.path.basename(path).split(".")[0]
        with open(path, "r") as f:
            cam_info = json.load(f)
            homography_inv = np.linalg.inv(cam_info.get("homography matrix"))
            cam_homographies[cam_id] = homography_inv
    
    area_info_path = os.path.join(info_dir, "AreaInfo", "area_info.json")
    with open(area_info_path, "r") as f:
        area_info = json.load(f)
    return cam_homographies, area_info

def calc_center(xmin, ymin, xmax, ymax, cam_homographies):
    if (xmin + xmax) / 2 < 1280 / 2:
        cam_id = "c0"
        center = np.array([(xmin + xmax) / 2, 
                            (ymin + ymax) / 2,
                            1])
    else:
        cam_id = "c1"
        center = np.array([(xmin + xmax) / 2 - 1280 / 2, 
                            (ymin + ymax) / 2,
                            1])
    center = cam_homographies[cam_id] @ center
    center = center / center[2]
    center = center[:2]
    return center

def which_area(center, area_info):
    for id, area in area_info.items():
        area = np.array(area)
        if center[0] >= area[0] and center[0] < area[2] and center[1] >= area[1] and center[1] < area[3]:
            return int(id)
    return -1

    
def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--camera-info-dir', type=str, default=None)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
