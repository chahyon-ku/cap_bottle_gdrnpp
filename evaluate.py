
import argparse
import collections
import sys
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import detectron2.config
import det.yolox.engine.yolox_trainer
import det.yolox.data.datasets.dataset_factory
import core.utils.my_checkpoint
import det.yolox.utils.boxes

import mmcv
import core.gdrn_modeling.models.GDRN_double_mask
import core.gdrn_modeling.datasets.data_loader
import core.gdrn_modeling.engine.gdrn_evaluator
import core.utils.data_utils

import cap_dataset


def add_arguments(parser):
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--yolox_model', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/output/yolox/bop_pbr/yolox_cap_bottle_10k/model_final.pth')
    parser.add_argument('--yolox_config', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/configs/yolox/bop_pbr/yolox_cap_bottle_10k.py')
    parser.add_argument('--gdrn_model', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/output/gdrn/cap_bottle/convnext_10k/model_final.pth')
    parser.add_argument('--gdrn_config', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/configs/gdrn/cap_bottle/convnext_10k.py')
    parser.add_argument('--data_dir', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/datasets/cap_bottle/test_1k')
    parser.add_argument('--output_dir', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/viz/all/test_1k')

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolox_cfg = detectron2.config.LazyConfig.load(args.yolox_config)
    yolox_cfg = detectron2.config.LazyConfig.apply_overrides(yolox_cfg, [f'train.init_checkpoint={args.yolox_model}'])
    det.yolox.data.datasets.dataset_factory.register_datasets_in_cfg(yolox_cfg)
    yolox_model = det.yolox.engine.yolox_trainer.YOLOX_DefaultTrainer.build_model(yolox_cfg)
    core.utils.my_checkpoint.MyCheckpointer(yolox_model, save_dir=yolox_cfg.train.output_dir).resume_or_load(yolox_cfg.train.init_checkpoint)
    # result = det.yolox.engine.yolox_trainer.YOLOX_DefaultTrainer.test(yolox_cfg, yolox_model)
    # print(result)

    gdrn_cfg = mmcv.Config.fromfile(args.gdrn_config)
    gdrn_cfg.merge_from_dict({'MODEL.WEIGHTS': args.gdrn_model, 'INPUT.WITH_DEPTH': True})
    if gdrn_cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(gdrn_cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(gdrn_cfg.SOLVER.OPTIMIZER_CFG)
            gdrn_cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = gdrn_cfg.SOLVER.OPTIMIZER_CFG
        gdrn_cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        gdrn_cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        gdrn_cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        gdrn_cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)

    gdrn_model, _ = core.gdrn_modeling.models.GDRN_double_mask.build_model_optimizer(gdrn_cfg, True)
    core.utils.my_checkpoint.MyCheckpointer(gdrn_model, save_dir=gdrn_cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(gdrn_cfg.MODEL.WEIGHTS)
    
    data_cfg = cap_dataset.CapDatasetConfig()
    data_cfg.data_dir = args.data_dir
    dataset = cap_dataset.CapDataset(data_cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    results = []
    with torch.no_grad():
        yolox_model.eval()
        gdrn_model.eval()
        for i_batch, batch_data in enumerate(dataloader):
            yolox_outputs = yolox_model(batch_data.rgb_yolo.cuda())['det_preds']
            yolox_outputs = det.yolox.utils.boxes.postprocess(yolox_outputs, 2)
            bbox_xyxy = yolox_outputs[0][:, :4]
            roi_classes = yolox_outputs[0][:, 6].long()
            scores = yolox_outputs[0][:, 4]

            order = torch.argsort(roi_classes, dim=-1, descending=False)
            bbox_xyxy = bbox_xyxy[order]
            roi_classes = roi_classes[order]
            roi_extents = torch.from_numpy(np.array([0.08, 0.08, 0.08])).cuda()

            coord_2d = core.utils.data_utils.get_2d_coord_np(batch_data.rgb.shape[2], batch_data.rgb.shape[1]).transpose(1, 2, 0)
            for i_obj in range(2):
                roi_whs = bbox_xyxy[i_obj, 2:] - bbox_xyxy[i_obj, :2]
                roi_centers = (bbox_xyxy[i_obj, 2:] + bbox_xyxy[i_obj, :2]) / 2
                scale = torch.max(roi_whs, dim=-1)[0] * 1.5
                roi_cams = batch_data.cam[0].cuda()
                resize_ratios = 64 / scale
                roi_coord_2d = core.utils.data_utils.crop_resize_by_warp_affine(coord_2d, roi_centers.cpu().numpy(), scale.item(), (64, 64)).transpose(2, 0, 1)
                roi_coord_2d_rel = (
                    roi_centers.reshape(2, 1, 1).cpu().numpy() - roi_coord_2d * np.array([640, 480]).reshape(2, 1, 1)
                ) / scale.cpu().numpy()

                roi_img = core.utils.data_utils.crop_resize_by_warp_affine(
                    batch_data.rgb[0].cpu().numpy(), roi_centers.cpu().numpy(), scale.item(), (256, 256), interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1).astype(np.float32) / 255
                gdrn_outputs = gdrn_model(
                    torch.from_numpy(roi_img).cuda()[None, ...],
                    roi_classes=roi_classes[None, i_obj],
                    roi_cams=roi_cams[None, ...],
                    roi_whs=roi_whs[None, ...],
                    roi_centers=roi_centers.cuda()[None, ...],
                    resize_ratios=resize_ratios[None, None],
                    roi_coord_2d=torch.from_numpy(roi_coord_2d).cuda()[None, ...],
                    roi_coord_2d_rel=torch.from_numpy(roi_coord_2d_rel).cuda()[None, ...],
                    roi_extents=roi_extents,
                )
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                axs[0, 0].imshow(batch_data.rgb[0].cpu().numpy())
                axs[0, 1].imshow(gdrn_outputs['full_mask'][0, 0].cpu())
                axs[0, 2].imshow(gdrn_outputs['region'][0, 0].cpu())
                axs[1, 0].imshow(gdrn_outputs['coor_x'][0, 0].cpu())
                axs[1, 1].imshow(gdrn_outputs['coor_y'][0, 0].cpu())
                axs[1, 2].imshow(gdrn_outputs['coor_z'][0, 0].cpu())
                cam_R_obj = gdrn_outputs['rot'][0].cpu().numpy()
                cam_t_obj = gdrn_outputs['trans'][0].cpu().numpy().reshape(3, 1)
                axis = 0.08 * np.array([[-0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, -0.5], [-0.5, -0.5, -0.5, 0.5]], dtype=float)
                axis = batch_data.cam[0].numpy() @ (cam_R_obj @ axis + cam_t_obj)
                axis = axis[:2] / axis[2]
                axs[0, 0].plot(axis[0, [0, 1]], axis[1, [0, 1]], 'r')
                axs[0, 0].plot(axis[0, [0, 2]], axis[1, [0, 2]], 'g')
                axs[0, 0].plot(axis[0, [0, 3]], axis[1, [0, 3]], 'b')

                os.makedirs(args.output_dir, exist_ok=True)
                plt.savefig(os.path.join(args.output_dir, f'{i_batch}_{i_obj}.png'))
                plt.close()

                # rgb_path = batch_data.rgb_path[0]
                # i_sample = int(os.path.basename(os.path.join(os.path.dirname(rgb_path), '../')))
                # i_frame = int(os.path.basename(rgb_path).split('.')[0])
                # # scene_id,im_id,obj_id,score,R,t,time
                # results.append(f'{i_sample}, {i_frame}, {i_obj + 1}, {scores[i_obj].item()}, {cam_R_obj.flatten().tolist()}, {cam_t_obj.flatten().tolist()}, {time.time()}\n')
    
    with open('results.csv', 'w') as f:
        f.write('scene_id,im_id,obj_id,score,R,t,time')
        f.writelines(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)