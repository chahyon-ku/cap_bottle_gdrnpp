import json
import os
import cv2

def main():
    bop_path = 'output/yolox/bop_pbr/yolox_cap_bottle_5k/inference/test_100/coco_instances_results_bop.json'

    with open(bop_path, 'r') as f:
        data = json.load(f)
    
    test = {}
    for item in data:
        key = '{}/{}'.format(item['scene_id'], item['image_id'])
        if key not in test:
            test[key] = {}
        if item['category_id'] not in test[key] or test[key][item['category_id']]['score'] < item['score']:
            test[key][item['category_id']] = {'bbox_est': item['bbox'], 'obj_id': item['category_id'], 'score': item['score'], 'time': item['time']}

    for key, value in test.items():
        test[key] = list(value.values())

    pred_path = bop_path.replace('.json', '_test.json')

    with open(pred_path, 'w') as f:
        print(test)
        json.dump(test, f, indent=1)

    data_dir = 'datasets/cap_bottle/test_100'
    output_dir = 'output/yolox/bop_pbr/yolox_cap_bottle_5k/viz/test_100'

    with open(pred_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    for key, value in data.items():
        scene_id, image_id = key.split('/')
        with open(os.path.join(data_dir, f'{int(scene_id):06d}/scene_gt_info.json'), 'r') as f:
            scene_gt_info = json.load(f)
        image_path = f'{data_dir}/{int(scene_id):06d}/rgb/{int(image_id):06d}.jpg'
        image = cv2.imread(image_path)
        for item in value:
            bbox = item['bbox_est']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(image, '{}'.format('cap' if item['obj_id'] == 1 else 'bottle'), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i_item, item in enumerate(scene_gt_info[image_id]):
            bbox = item['bbox_obj']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(image, '{}'.format('cap' if i_item == 0 else 'bottle'), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('image', image)
        cv2.imwrite(f'{output_dir}/{scene_id}_{image_id}.png', image)

if __name__ == '__main__':
    main()