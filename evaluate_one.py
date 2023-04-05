import argparse
import numpy as np
import cv2
import torch
import det.yolox.data.datasets.dataset_factory
import det.yolox.engine.yolox_trainer
import core.utils.my_checkpoint
import core.gdrn_modeling.models.GDRN_double_mask
import detectron2.config
import mmcv
import os

def add_arguments(parser):
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--yolox_model', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/output/yolox/bop_pbr/yolox_cap_bottle_10k/model_final.pth')
    parser.add_argument('--yolox_config', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/configs/yolox/bop_pbr/yolox_cap_bottle_10k.py')
    parser.add_argument('--gdrn_model', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/output/gdrn/cap_bottle/convnext_10k/model_final.pth')
    parser.add_argument('--gdrn_config', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/configs/gdrn/cap_bottle/convnext_10k.py')
    parser.add_argument('--output_dir', type=str, default='/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/viz/all/real')

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

    rgb = cv2.imread('../experiments/output/color_000000.png')
    rgb_yolo, _ = det.yolox.data.data_augment.preproc(rgb, (640, 640))
    cam = np.array(open('../experiments/output/K_000000.txt').read().split()).astype(np.float32).reshape(3, 3)

    yolox_model.eval()
    gdrn_model.eval()
    with torch.no_grad():
        det_preds = yolox_model(torch.from_numpy(rgb_yolo).cuda().unsqueeze(0))['det_preds']
        yolox_outputs = det.yolox.utils.boxes.postprocess(yolox_outputs, 2)
        bbox_xyxy = yolox_outputs[0][:, :4]
        scores = yolox_outputs[0][:, 4]
        roi_classes = yolox_outputs[0][:, 6].long()
        print(det_preds)
        for i_obj in range(2):
            roi_centers = (bbox_xyxy[i_obj, 2:] + bbox_xyxy[i_obj, :2]) / 2
            roi_img = core.utils.data_utils.crop_resize_by_warp_affine(
                rgb, roi_centers.cpu().numpy(), scale.item(), (256, 256), interpolation=cv2.INTER_LINEAR
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
