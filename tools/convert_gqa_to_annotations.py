import os
import json
import argparse

def process_and_dump(gqa_vg, args, split='train'):
    # with open(os.path.join(args.splits, '%s_clean.json'%split)) as f:
    #     idx = json.load(f)

    idx = sorted(list(gqa_vg.keys()))
    # split_gqa_vg = {img_id:gqa_vg[str(img_id)] for img_id in idx if str(img_id) in gqa_vg}
    # print ('Out of ', len(idx), 'images, found ', len(split_gqa_vg))

    for img_id, record in gqa_vg.items():
        record.pop('location', None)
        record.pop('weather', None)
        objects = []
        for obj_id, obj in record['objects'].items():
            obj.pop('attributes', None)
            obj.pop('relations', None)
            objects.append(obj)
        record['objects'] = objects

    with open(os.path.join(args.destination, '%s_annotations.json'%split), 'w+') as f:
        json.dump(gqa_vg, f)

    with open(os.path.join(args.destination, '%s_clean.json'%split), 'w+') as f:
            json.dump(idx, f)    

    if args.mini_version:
        mini_gqa_vg = {key: value for index, (key, value) in enumerate(gqa_vg.items()) if index < 1000}
        with open(os.path.join(args.destination, 'mini%s_annotations.json'%split), 'w+') as f:
            json.dump(mini_gqa_vg, f)
        keys = list(mini_gqa_vg.keys())
        with open(os.path.join(args.destination, 'mini%s_clean.json'%split), 'w+') as f:
            json.dump(keys, f)

def main(args):
    assert os.path.isfile(args.gqa_vg_train)
    assert os.path.isfile(args.gqa_vg_val)
    assert os.path.isdir(args.destination)

    with open(args.gqa_vg_train) as f:
        gqa_vg_train = json.load(f)

    with open(args.gqa_vg_val) as f:
        gqa_vg_val = json.load(f)

    object_labels = set()
    for idx, record in gqa_vg_train.items():
        for obj_key, obj in record['objects'].items():
            object_labels.add(obj['name'])

    for idx, record in gqa_vg_val.items():
        for obj_key, obj in record['objects'].items():
            object_labels.add(obj['name'])

    object_labels = sorted(list(object_labels))
    with open(os.path.join(args.destination, 'objects.json'), 'w+') as f:
            json.dump(object_labels, f)

    val_test_keys = sorted(list(gqa_vg_val.keys()))

    gqa_vg_test = {y: gqa_vg_val[y] for y in test_keys}
    gqa_vg_val = {x: gqa_vg_val[x] for x in val_keys}

    process_and_dump(gqa_vg_train, args, 'train')
    process_and_dump(gqa_vg_val, args, 'val')
    process_and_dump(gqa_vg_test, args, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gqa_vg_train',
        default='../data/gqa/train_sceneGraphs.json',
        type=str,
        help='Path to GQAs visual genome TRAIN dataset json file')
    parser.add_argument('--gqa_vg_val',
        default='../data/gqa/val_sceneGraphs.json',
        type=str,
        help='Path to GQAs visual genome VAL dataset json file')
    parser.add_argument('--destination',
        default='data/VG',
        type=str,
        help='Path where the annotations will be dumped')
    parser.add_argument('--mini_version',
        action='store_true',
        help='Creates a mini version of the dataset for fast debugging')

    args = parser.parse_args()
    main(args)
