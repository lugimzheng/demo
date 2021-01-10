import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes, draw_ID
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from pose.pose_estimation import pose_estimate
from fall_down_detection import fall
from face_recognition import who
from face import face_recognition


class VideoTracker(object):
    def __init__(self, cfg, args, video_path, face_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        #self.video_path_depth = video_path_depth
        self.logger = get_logger("root")
        self.face_path = face_path

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
            #self.vdo_depth = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "RGB Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

            #assert os.path.isfile(self.video_path_depth), "depth Path error"
            #self.vdo_depth.open(self.video_path_depth)
            #assert self.vdo_depth.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 15, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        names = {}
        dists_min = {}
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            #_, ori_im_depth = self.vdo_depth.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            #im_depth = cv2.cvtColor(ori_im_depth, cv2.COLOR_BGR2RGB)
            
            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2

            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]

                im_t = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                for bb_xyxy, id in zip(bbox_xyxy, identities):
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                    x1, y1, x2, y2 = [int(i) for i in bb_xyxy]
                    im_person = im_t[y1:y2, x1:x2, :]

                    # face recognition
                    name, dist_min = who(im_person, self.face_path)
                    if not (id in names):
                        names[id] = name
                        dists_min[id] = dist_min
                    else:
                        if dists_min[id] > dist_min:
                            names[id] = name
                            dists_min[id] = dist_min

                    im_person = im_t[y1:y2, x1:x2, :]/255

                    # pose
                    pose, colors, pairs, colors_skeleton = pose_estimate(im_person)
                    
                    pose += np.array([x1, y1])
                    pose = pose.astype(np.int)
                    
                    for idx in range(len(colors)):
                        cv2.circle(ori_im, (pose[idx, 0], pose[idx, 1]), 3, colors[idx], thickness=3, lineType=8,
                                   shift=0)

                    for idx in range(len(colors_skeleton)):
                        ori_im = cv2.line(ori_im, (pose[pairs[idx][0], 0], pose[pairs[idx][0], 1]),
                                          (pose[pairs[idx][1], 0], pose[pairs[idx][1], 1]), colors_skeleton[idx], 3)

                    # fall down detection
                    is_fall = fall(pose)
                    if is_fall:
                        t_size = cv2.getTextSize('fall', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                        cv2.putText(ori_im, 'fall', (x1, y1 - t_size[1] - 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                    [0, 0, 255],2)

                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, names)
                #ori_im = draw_ID(ori_im, im_depth, bbox_xyxy, pose)

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)  # RGB video path
    # parser.add_argument("VIDEO_PATH_Depth", type=str)  # Depth video path
    parser.add_argument("FACE_DATABASE", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH,
                      face_path=args.FACE_DATABASE) as vdo_trk:
        vdo_trk.run()
