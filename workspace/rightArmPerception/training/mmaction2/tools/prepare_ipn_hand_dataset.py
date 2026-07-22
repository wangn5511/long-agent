"""
This file is used to prepare the IPN Hand dataset for training and testing. 
In the framework of mmaction2
The dataset to be used is RawframeDataset
"""

import argparse
import os
import os.path as osp
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--frame_root_path", type=str, required=True)
parser.add_argument("--jester_to_ipn_mapping_file", type=str, required=True, 
    help="The label mapping file from Jester to IPN; only these mapped label will be used")
parser.add_argument("--jester_label_set_file", type=str, required=True, 
    help="The label set file of the Jester dataset")
parser.add_argument("--anno_list_file", type=str, required=True, 
    help="The annotation list file of the IPN Hand dataset")
parser.add_argument("--split_file", type=str, required=True, 
    help="The split file of the IPN Hand dataset")
parser.add_argument("--out_ann_file", type=str, required=True, 
    help="The output annotation file of the IPN Hand dataset")
parser.add_argument("--class_detail_file", type=str, required=True, 
    help="The class detail file of the IPN Hand dataset")
args = parser.parse_args()


def prepare_ipn_hand_dataset():
    with open(args.jester_label_set_file, "r") as f:
        lines = f.readlines()
    jester_label_set = []
    for line in lines:
        jester_label_set.append(line.strip())

    with open(args.jester_to_ipn_mapping_file, "r") as f:
        lines = f.readlines()
    ipn_to_jester_mapping = {}
    for line in lines:
        jester_label, ipn_label = line.strip().split(";")
        ipn_to_jester_mapping[ipn_label] = jester_label

    with open(args.class_detail_file, "r") as f:
        lines = f.readlines()
    class_detail_mapping = {}
    for line in lines[1:]:
        infos = line.strip().split("\t")
        class_name_str = infos[-1]
        class_name_str = class_name_str.lstrip()
        class_nick_name = infos[1].lstrip().rstrip()
        class_detail_mapping[class_nick_name] = class_name_str

    with open(args.split_file, "r") as f:
        lines = f.readlines()
    video_name_list = []
    for line in lines:
        video_name, _ = line.strip().split("\t")
        video_name = video_name.rstrip()
        video_name_list.append(video_name)

    print("Got {} videos in the split file".format(len(video_name_list)))

    label_count = {}

    with open(args.anno_list_file, "r") as f:
        lines = f.readlines()
    video_name_to_offset_mapping = {}
    for line in lines[1:]:
        infos = line.strip().split(",")
        video_name = infos[0]
        label_nick_name = infos[1]
        label_real_name = class_detail_mapping[label_nick_name]

        if label_real_name not in ipn_to_jester_mapping:
            continue

        if label_real_name not in label_count:
            label_count[label_real_name] = 0
        label_count[label_real_name] += 1

        start_frame_idx = int(infos[3]) - 1  # this is 1 based
        end_frame_idx = int(infos[4]) - 1  # this is 1 based
        num_frames = int(infos[5])
        assert num_frames == end_frame_idx - start_frame_idx + 1, \
            "The number of frames is not correct"
        video_name_to_offset_mapping[video_name] = (start_frame_idx, num_frames, label_real_name)

    print("Got {} videos in the annotation list file".format(len(video_name_to_offset_mapping)))

    print("Label count:")
    for label_real_name, count in label_count.items():
        print("{}: {}".format(label_real_name, count))

    outlines = []
    for video_name in video_name_list:
        offset_info = video_name_to_offset_mapping[video_name]
        start_frame_idx, num_frames, ipn_label_real_name = offset_info
        jester_label = ipn_to_jester_mapping[ipn_label_real_name]
        mapped_label_id = jester_label_set.index(jester_label)

        relative_path = osp.join(args.frame_root_path, video_name)

        outlines.append(f"{relative_path} {start_frame_idx} {num_frames} {mapped_label_id}\n")

    with open(args.out_ann_file, "w") as f:
        f.writelines(outlines)


if __name__ == "__main__":
    prepare_ipn_hand_dataset()
