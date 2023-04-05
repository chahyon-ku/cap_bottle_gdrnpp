import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np


def main(args):
    preds = {}
    with open(args.pred_path, 'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            words = line.strip().split(',')
            assert len(words) == 7
            scene_id = int(words[0])
            image_id = int(words[1])
            obj_id = int(words[2])
            score = float(words[3])
            R = np.array(list(map(float, words[4].split(' ')))).reshape(3, 3)
            t = np.array(list(map(float, words[5].split(' ')))).reshape(3, 1)
            time = words[6]
            key = f'{scene_id}/{image_id}'
            print(key)
            if key not in preds:
                preds[key] = {}
            preds[key][obj_id] = {'obj_id': obj_id, 'score': score, 'cam_R_m2c': R, 'cam_t_m2c': t, 'time': time}

            line = f.readline()

    scene_dirs = sorted(os.listdir(args.input_dir))
    for scene_dir in scene_dirs:
        if len(scene_dir) != 6 or not os.path.isdir(os.path.join(args.input_dir, scene_dir)):
            continue
        scene_gt_path = os.path.join(args.input_dir, scene_dir, 'scene_gt.json')
        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)
        scene_camera_path = os.path.join(args.input_dir, scene_dir, 'scene_camera.json')
        with open(scene_camera_path, 'r') as f:
            scene_camera = json.load(f)

        rgb_names = os.listdir(os.path.join(args.input_dir, scene_dir, 'rgb'))
        rgb_names = sorted(rgb_names)
        for rgb_name in rgb_names:
            i_frame = int(rgb_name.split('.')[0])
            rgb_path = os.path.join(args.input_dir, scene_dir, 'rgb', rgb_name)
            depth_path = os.path.join(args.input_dir, scene_dir, 'depth', f'{i_frame:06d}.png')
            mask0_path = os.path.join(args.input_dir, scene_dir, 'mask', f'{i_frame:06d}_{0:06d}.png')
            mask1_path = os.path.join(args.input_dir, scene_dir, 'mask', f'{i_frame:06d}_{1:06d}.png')
            mask_visib0_path = os.path.join(args.input_dir, scene_dir, 'mask_visib', f'{i_frame:06d}_{0:06d}.png')
            mask_visib1_path = os.path.join(args.input_dir, scene_dir, 'mask_visib', f'{i_frame:06d}_{1:06d}.png')

            cam_K = np.array(scene_camera[f'{i_frame}']['cam_K']).reshape(3, 3)
            cam_R_m2c = np.array(scene_gt[f'{i_frame}'][0]['cam_R_m2c']).reshape(3, 3)
            cam_t_m2c = np.array(scene_gt[f'{i_frame}'][0]['cam_t_m2c']).reshape(3, 1)
            points = 80 * np.array([[-0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, -0.5], [-0.5, -0.5, -0.5, 0.5]], dtype=float)
            points = cam_K @ (cam_R_m2c @ points + cam_t_m2c)
            points = points / points[2]

            cam_R_m2c = np.array(scene_gt[f'{i_frame}'][1]['cam_R_m2c']).reshape(3, 3)
            cam_t_m2c = np.array(scene_gt[f'{i_frame}'][1]['cam_t_m2c']).reshape(3, 1)
            points1 = 80 * np.array([[-0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, -0.5], [-0.5, -0.5, -0.5, 0.5]], dtype=float)
            points1 = cam_K @ (cam_R_m2c @ points1 + cam_t_m2c)
            points1 = points1 / points1[2]
            
            # fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(18, 9))
            plt.figure(figsize=(12, 9))
            plt.imshow(plt.imread(rgb_path))
            plt.plot(points[0, [0, 1]], points[1, [0, 1]], 'r')
            plt.plot(points[0, [0, 2]], points[1, [0, 2]], 'g')
            plt.plot(points[0, [0, 3]], points[1, [0, 3]], 'b')
            plt.plot(points1[0, [0, 1]], points1[1, [0, 1]], 'r')
            plt.plot(points1[0, [0, 2]], points1[1, [0, 2]], 'g')
            plt.plot(points1[0, [0, 3]], points1[1, [0, 3]], 'b')

            key = f'{int(scene_dir)}/{i_frame}'
            if key in preds:
                for obj_id in preds[key]:
                    pred = preds[key][obj_id]
                    R = pred['cam_R_m2c']
                    t = pred['cam_t_m2c']
                    # print(cam_R_m2c, R, cam_t_m2c, t)
                    points2 = 80 * np.array([[-0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, -0.5], [-0.5, -0.5, -0.5, 0.5]], dtype=float)
                    points2 = cam_K @ (R @ points2 + t)
                    points2 = points2 / points2[2]
                    plt.plot(points2[0, [0, 1]], points2[1, [0, 1]], 'm')
                    plt.plot(points2[0, [0, 2]], points2[1, [0, 2]], 'y')
                    plt.plot(points2[0, [0, 3]], points2[1, [0, 3]], 'c')
            else:
                print(key)

            plt.title(f'Frame {i_frame}')
            # axs[0, 1].imshow(plt.imread(mask0_path))
            # axs[0, 1].set_title('Mask 0')
            # axs[0, 2].imshow(plt.imread(mask1_path))
            # axs[0, 2].set_title('Mask 1')
            # axs[1, 0].imshow(plt.imread(depth_path), cmap='gray')
            # axs[1, 0].set_title('Depth')
            # axs[1, 1].imshow(plt.imread(mask_visib0_path))
            # axs[1, 1].set_title('Mask visib 0')
            # axs[1, 2].imshow(plt.imread(mask_visib1_path))
            # axs[1, 2].set_title('Mask visib 1')
            plt.savefig(os.path.join(args.output_dir, f'{scene_dir}_{i_frame:06d}.png'), bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/cap_bottle/test_1k')
    parser.add_argument('--pred_path', type=str, default='output/gdrn/cap_bottle/convnext_10k/inference_model_final/test_1k/convnext-10kRANSAC-PNP-test-iter0_cap_bottle-test.csv')
    parser.add_argument('--output_dir', type=str, default='viz/gdrn/test_1k')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
