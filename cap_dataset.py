import torch
import typing
import os
import json
import numpy as np
import tqdm
import cv2
import det.yolox.data.data_augment
import detectron2.data.transforms


class CapDatasetConfig:
    data_dir = ''

Point2D = typing.NamedTuple('Point2D', [('x', int), ('y', int)])

class CapObjectInfo:
    mask_path = ''
    mask_visib_path = ''
    bbox_top_left: Point2D = []
    bbox_bottom_right: Point2D = []
    cam_R_obj = []
    cam_t_obj = []

class CapSampleInfo:
    rgb_path = ''
    depth_path = ''
    cam = []
    depth_scale = 0.0
    objects: typing.List[CapObjectInfo] = []

CapSample = typing.NamedTuple('CapSample', [('rgb', np.ndarray), ('rgb_yolo', np.ndarray), ('rgb_path', str),
                                            ('cam', np.ndarray), ('depth_scale', float),
                                            ('masks', np.ndarray), ('masks_visibs', np.ndarray),
                                            ('bboxes_top_left', np.ndarray), ('bboxes_bottom_right', np.ndarray),
                                            ('cam_R_objs', np.ndarray), ('cam_t_objs', np.ndarray)])

class CapDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg: CapDatasetConfig):
        self.data_cfg = data_cfg
        self.data_entries: typing.List[CapSampleInfo] = []

        scene_dirs = sorted(os.listdir(self.data_cfg.data_dir))
        scene_dirs = filter(lambda x: os.path.isdir(os.path.join(self.data_cfg.data_dir, x)), scene_dirs)
        scene_dirs = filter(lambda x: os.path.exists(os.path.join(self.data_cfg.data_dir, x, 'scene_camera.json')), scene_dirs)
        scene_dirs = list(scene_dirs)
        for scene_dir in tqdm.tqdm(scene_dirs, 'Loading Cap Dataset', total=len(scene_dirs), unit='scenes'):
            rgb_dir = os.path.join(self.data_cfg.data_dir, scene_dir, 'rgb')
            depth_dir = os.path.join(self.data_cfg.data_dir, scene_dir, 'depth')
            mask_dir = os.path.join(self.data_cfg.data_dir, scene_dir, 'mask')
            mask_visib_dir = os.path.join(self.data_cfg.data_dir, scene_dir, 'mask_visib')
            with open(os.path.join(self.data_cfg.data_dir, scene_dir, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(self.data_cfg.data_dir, scene_dir, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.data_cfg.data_dir, scene_dir, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)
            
            for frame_idx in range(len(scene_camera)):
                frame_data = CapSampleInfo()
                frame_data.rgb_path = os.path.join(rgb_dir, '%06d.jpg' % (frame_idx))
                frame_data.depth_path = os.path.join(depth_dir, '%06d.png' % (frame_idx))
                frame_data.cam = scene_camera[str(frame_idx)]['cam_K']
                frame_data.depth_scale = scene_camera[str(frame_idx)]['depth_scale']
                frame_data.objects = []
                for obj_idx in range(len(scene_gt[str(frame_idx)])):
                    obj_data = CapObjectInfo()
                    obj_data.mask_path = os.path.join(mask_dir, '%06d_%06d.png' % (frame_idx, obj_idx))
                    obj_data.mask_visib_path = os.path.join(mask_visib_dir, '%06d_%06d.png' % (frame_idx, obj_idx))
                    obj_data.bbox_top_left = Point2D(x=scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][0],
                                                     y=scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][1])
                    obj_data.bbox_bottom_right = Point2D(x=scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][0] + scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][2],
                                                         y=scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][1] + scene_gt_info[str(frame_idx)][obj_idx]['bbox_obj'][3])
                    obj_data.cam_R_obj = scene_gt[str(frame_idx)][obj_idx]['cam_R_m2c']
                    obj_data.cam_t_obj = scene_gt[str(frame_idx)][obj_idx]['cam_t_m2c']
                    frame_data.objects.append(obj_data)
                self.data_entries.append(frame_data)
        
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx) -> CapSample:
        rgb = cv2.imread(self.data_entries[idx].rgb_path)
        rgb_yolo, _ = det.yolox.data.data_augment.preproc(rgb, (640, 640))
        depth = cv2.imread(self.data_entries[idx].depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 10000
        cam = np.array(self.data_entries[idx].cam).reshape((3, 3))
        depth_scale = self.data_entries[idx].depth_scale
        masks = np.stack([cv2.imread(obj_info.mask_path, cv2.IMREAD_ANYDEPTH) for obj_info in self.data_entries[idx].objects], axis=0)
        masks_visibs = np.stack([cv2.imread(obj_info.mask_visib_path, cv2.IMREAD_ANYDEPTH) for obj_info in self.data_entries[idx].objects], axis=0)
        bboxes_top_left = np.stack([obj_info.bbox_top_left for obj_info in self.data_entries[idx].objects], axis=0)
        bboxes_bottom_right = np.stack([obj_info.bbox_bottom_right for obj_info in self.data_entries[idx].objects], axis=0)
        cam_R_objs = np.stack([obj_info.cam_R_obj for obj_info in self.data_entries[idx].objects], axis=0)
        cam_t_objs = np.stack([obj_info.cam_t_obj for obj_info in self.data_entries[idx].objects], axis=0)
        cap_sample = CapSample(rgb=rgb, rgb_yolo=rgb_yolo, depth=depth, cam=cam, rgb_path = self.data_entries[idx].rgb_path,
                               depth_scale=depth_scale, masks=masks, masks_visibs=masks_visibs,
                               bboxes_top_left=bboxes_top_left, bboxes_bottom_right=bboxes_bottom_right,
                               cam_R_objs=cam_R_objs, cam_t_objs=cam_t_objs)
        return cap_sample