import argparse
import os
import pickle
import os.path as osp
import glob

import numpy as np
from sklearn.metrics import confusion_matrix


def check_results():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--jester_label_set_file", type=str, required=True)
    parser.add_argument("--jester_to_ipn_mapping_file", type=str, required=True, 
    help="The label mapping file from Jester to IPN; only these mapped label will be used")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--anno_list_file", type=str, required=True, 
        help="The annotation list file of the IPN Hand dataset")
    parser.add_argument("--class_detail_file", type=str, required=True, 
        help="The class detail file of the IPN Hand dataset")
    args = parser.parse_args()

    with open(args.prediction_file, "rb") as f:
        predictions = pickle.load(f)
    print("Got {} predictions".format(len(predictions)))

    pred_list = []
    gt_list = []
    for data in predictions:
        pred = data["pred_label"]
        gt = data["gt_label"]
        pred_list.append(pred.item())
        gt_list.append(gt.item())

    with open(args.jester_label_set_file, "r") as f:
        lines = f.readlines()
    jester_label_set = []
    for line in lines:
        jester_label_set.append(line.strip())

    with open(args.ann_file, "r") as f:
        lines = f.readlines()
    video_name_list = []
    offset_list = []
    for line in lines:
        infos = line.strip().split(" ")
        video_name = infos[0].split("/")[-1]
        video_name_list.append(video_name)
        offset_list.append(int(infos[1]) + 1)

    with open(args.class_detail_file, "r") as f:
        lines = f.readlines()
    class_detail_mapping = {}
    for line in lines[1:]:
        infos = line.strip().split("\t")
        class_name_str = infos[-1]
        class_name_str = class_name_str.lstrip()
        class_nick_name = infos[1].lstrip().rstrip()
        class_detail_mapping[class_nick_name] = class_name_str

    with open(args.jester_to_ipn_mapping_file, "r") as f:
        lines = f.readlines()
    ipn_to_jester_mapping = {}
    for line in lines:
        jester_label, ipn_label = line.strip().split(";")
        ipn_to_jester_mapping[ipn_label] = jester_label

    # val_real_name_list = []
    # with open(args.anno_list_file, "r") as f:
    #     lines = f.readlines()
    # for line in lines[1:]:
    #     infos = line.strip().split(",")
    #     video_name = infos[0]
    #     label_nick_name = infos[1]
    #     label_real_name = class_detail_mapping[label_nick_name]
    #     if video_name not in video_name_list:
    #         continue
    #     if label_real_name not in ipn_to_jester_mapping:
    #         continue
    #     val_real_name_list.append(label_real_name)

    # import pdb; pdb.set_trace()
    # assert len(val_real_name_list) == len(video_name_list)

    for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        # if gt == 25:
        #     continue
        if pred != gt:
            print(f"path: {video_name_list[i]} \
                shift: {offset_list[i]} \
                    jester pred: {jester_label_set[pred]} \
                        jester gt label: {jester_label_set[gt]}")

    # pred_array = np.array(pred_list)
    # gt_array = np.array(gt_list)

    # label_set = sorted(set(gt_array))

    # pred_set = sorted(set(pred_array))

    # print(label_set)

    # confusion_mat = confusion_matrix(gt_array, pred_array, labels=np.arange(27))
    # print(confusion_mat)

    # print("pred set: {}".format(pred_set))


if __name__ == "__main__":
    check_results()