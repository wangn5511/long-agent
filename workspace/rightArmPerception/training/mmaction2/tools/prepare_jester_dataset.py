"""
This file is used to prepare the Jester dataset for training and testing. 
In the framework of mmaction2
The dataset to be used is RawframeDataset
"""

import argparse
import os
import os.path as osp
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--raw_csv_file", type=str, required=True)
parser.add_argument("--raw_label_set_file", type=str, required=True)
parser.add_argument("--out_ann_file", type=str, required=True)
args = parser.parse_args()


def prepare_jester_dataset():

    with open(args.raw_label_set_file, "r") as f:
        lines = f.readlines()
    label_set = []
    for line in lines:
        label_set.append(line.strip())

    with open(args.raw_csv_file, "r") as f:
        lines = f.readlines()

    outlines = []
    for line in lines:
        relative_path, label_str = line.strip().split(";")
        video_path = osp.join(args.data_path, relative_path)
        imgs_list = glob.glob(osp.join(video_path, "*.jpg"))
        label_id = label_set.index(label_str)
        outlines.append(f"{relative_path} {len(imgs_list)} {label_id}\n")

    with open(args.out_ann_file, "w") as f:
        f.writelines(outlines)


if __name__ == "__main__":
    prepare_jester_dataset()
