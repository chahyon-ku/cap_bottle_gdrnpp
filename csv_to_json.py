import json
import csv
import numpy as np
import os


if __name__ == '__main__':
    csv_path = '/home/rpm/Lab/cap_bottle/cap_bottle_gdrnpp/output/gdrn/cap_bottle/convnext_10k/inference_model_final/test_1k/convnext-10kRANSAC-PNP-test-iter0_cap_bottle-test.csv'

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    json_data = {}
    for row in csv_data:
        if row[0] == 'scene_id':
            continue
        sample_id = int(row[0])
        frame_id = int(row[1])
        obj_id = int(row[2])
        score = float(row[3])
        R = [float(el) for el in row[4].split()]
        t = [float(el) for el in row[5].split()]

        if f'{sample_id}/{frame_id}' not in json_data:
            json_data[f'{sample_id}/{frame_id}'] = []
        json_data[f'{sample_id}/{frame_id}'].insert(obj_id - 1, {'obj_id': obj_id, 'score': score, 'cam_R_m2c': R, 'cam_t_m2c': t})

    with open(csv_path.replace('.csv', '.json'), 'w') as f:
        json.dump(json_data, f, indent=1)
    
