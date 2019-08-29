import os
import json
import argparse

def process_and_dump(gqa_vg, args, split='train'):
    with open(os.path.join(args.splits, '%s_clean.json'%split)) as f:
        idx = json.load(f)

    split_gqa_vg = {img_id:gqa_vg[img_id] for img_id in idx}

    for img_id, record in split_gqa_vg.items():
        record.pop('location', None)
        record.pop('weather', None)
        objects = []
        for obj_id, obj in record['objects'].items():
            obj.pop('attributes', None)
            obj.pop('relations', None)
            objects.append(obj)
        record['objects'] = objects

    with open(os.path.join(args.destination, '%s_annotations.json'), 'w+') as f:
        json.dump(split_gqa_vg, f)

    if args.mini_version:
        mini_gqa_vg = {key: value for index, (key, value) in enumerate(split_gqa_vg.items()) if index < 1000}
        with open(os.path.join(args.destination, 'mini%s_annotations.json'), 'w+') as f:
            json.dump(mini_gqa_vg, f)
        keys = list(mini_gqa_vg.keys())
        with open(os.path.join(args.destination, 'mini%s_clean.json'), 'w+') as f:
            json.dump(keys, f)

    return split_gqa_vg

def main(args):
    assert os.path.isfile(args.gqa_vg)
    assert os.path.isdir(args.splits)
    assert os.path.isdir(args.destination)
    assert os.path.isfile(os.path.join(args.splits, 'train_clean.json'))
    assert os.path.isfile(os.path.join(args.splits, 'val_clean.json'))
    assert os.path.isfile(os.path.join(args.splits, 'test_clean.json'))

    with open(args.gqa_vg) as f:
        gqa_vg = json.load(f)

    train_gqa_vg = process(gqa_vg, args, 'train')
    val_gqa_vg = process(gqa_vg, args, 'val')
    test_gqa_vg = process(gqa_vg, args, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gqa_vg',
        default='/home/amrita/pytorch-faster-rcnn/data/VG/gqa_vg.json',
        type=str,
        help='Path to GQAs visual genome dataset json file')
    parser.add_argument('--splits',
        default='/home/amrita/pytorch-faster-rcnn/data/VG',
        type=str,
        help='Path to directory containing train-val-test split in json files')
    parser.add_argument('--destination',
        default='/home/amrita/pytorch-faster-rcnn/data/VG',
        type=str,
        help='Path where the annotations will be dumped')
    parser.add_argument('--mini_version',
        action='store_true',
        type=bool,
        help='Creates a mini version of the dataset for fast debugging')

    args = parser.parse_args()
    main(args)