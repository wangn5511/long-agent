from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--raw_frame_dataset_file", type=str, required=True)
parser.add_argument("--with_offset", action="store_true")
args = parser.parse_args()


old_class_mapping = {
    0: "NONE",
    1: "OK",
    2: "WAVE_IN",
    3: "WAVE_OUT",
    8: "BYEBYE",
    9: "THUMBS_UP",
}

new_class_mapping = {
    "NONE": 0,
    "OK": 1,
    "WAVE_IN": 2,
    "WAVE_OUT": 3,
    "BYEBYE": 4,
    "THUMBS_UP": 5,
}


def main():
    assert args.with_offset, "This script only supports with_offset=True"  # TODO:
    with open(args.raw_frame_dataset_file, "r") as f:
        lines = f.readlines()
    outlines = []
    for line in lines:
        infos = line.strip().split(' ')
        label_id = int(infos[-1])

        if label_id in old_class_mapping:
            new_label_id = new_class_mapping[old_class_mapping[label_id]]
        else:
            continue

        new_line = f"{infos[0]} {infos[1]} {infos[2]} {new_label_id}\n"
        outlines.append(new_line)

    new_dataset_file = Path(args.raw_frame_dataset_file).stem + "_only_6_classes.txt"
    # new_dataset_file = args.raw_frame_dataset_file.replace(".txt", "_only_6_classes.txt")
    with open(new_dataset_file, "w") as f:
        f.writelines(outlines)

    print(f"==> Written new dataset file to {new_dataset_file}")
    print(f"\tdataset length reduced from {len(lines)} to {len(outlines)}")


if __name__ == "__main__":
    main()
