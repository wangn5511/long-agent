#!/usr/bin/env python3
"""Recover video names for misclassified samples without re-running inference.

Both classified_test.txt and misclassified_test.txt are written in the same
loop that iterates over the val split in fixed order (shuffle=False).
By replaying that order against the pkl's val split we can unambiguously
assign a frame_dir / video_name to every line.

Usage:
    python handoff/stgcn_hand21_3d/tools/recover_misclassified_names.py \
        --pkl  /home/results/6t1/gesture_class9_3d/gesture_class9_hand21_3d.pkl \
        --work-dir /home/results/6t1/stgcn9_hand21_u16_3d_click_cont
"""
import argparse
import csv
import pickle
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--bad-file',  default='misclassified_test.txt')
    p.add_argument('--good-file', default='classified_test.txt')
    p.add_argument('--output',    default='misclassified_named.csv')
    return p.parse_args()


def main():
    args = parse_args()

    with args.pkl.open('rb') as f:
        data = pickle.load(f)

    categories = data['categories']
    # PoseDataset with shuffle=False processes val samples in split['val'] order.
    val_ids = data['split']['val']
    id2ann  = {a['frame_dir']: a for a in data['annotations']}

    bad_path  = args.work_dir / args.bad_file
    good_path = args.work_dir / args.good_file

    # Each line: None,pred_class,gt_class,score
    def read_lines(path):
        lines = []
        with path.open() as f:
            for row in f:
                parts = row.strip().split(',')
                if len(parts) < 4:
                    continue
                lines.append({'pred': parts[1], 'gt': parts[2], 'score': parts[3]})
        return lines

    bad_lines  = read_lines(bad_path)
    good_lines = read_lines(good_path)

    print(f'val split : {len(val_ids)} samples')
    print(f'classified: {len(good_lines)} lines')
    print(f'misclassified: {len(bad_lines)} lines')
    assert len(val_ids) == len(bad_lines) + len(good_lines), \
        "Line count mismatch — files may be from a different run than the pkl"

    bad_ptr = good_ptr = 0
    recovered_bad = []
    ambiguous = 0

    for val_id in val_ids:
        ann     = id2ann[val_id]
        gt_name = categories[ann['label']]

        next_bad_gt  = bad_lines[bad_ptr]['gt']   if bad_ptr  < len(bad_lines)  else None
        next_good_gt = good_lines[good_ptr]['gt']  if good_ptr < len(good_lines) else None

        bad_match  = (next_bad_gt  == gt_name)
        good_match = (next_good_gt == gt_name)

        if bad_match and not good_match:
            entry = bad_lines[bad_ptr]
            bad_ptr += 1
            recovered_bad.append({
                'frame_dir':  val_id,
                'video_name': ann.get('video_name', ''),
                'gt':         entry['gt'],
                'pred':       entry['pred'],
                'score':      entry['score'],
            })
        elif good_match and not bad_match:
            good_ptr += 1
        elif bad_match and good_match:
            # Both files have same gt at their cursor — ambiguous position.
            # Heuristic: pick based on pred==gt to distinguish.
            bad_pred_is_wrong  = (bad_lines[bad_ptr]['pred']  != gt_name)
            good_pred_is_right = (good_lines[good_ptr]['pred'] == gt_name)
            if bad_pred_is_wrong and good_pred_is_right:
                entry = bad_lines[bad_ptr]; bad_ptr += 1
                recovered_bad.append({
                    'frame_dir':  val_id,
                    'video_name': ann.get('video_name', ''),
                    'gt':         entry['gt'],
                    'pred':       entry['pred'],
                    'score':      entry['score'],
                })
            else:
                good_ptr += 1
                ambiguous += 1
        else:
            # Neither matches — data inconsistency
            print(f'  WARNING: no match for {val_id} gt={gt_name} '
                  f'next_bad_gt={next_bad_gt} next_good_gt={next_good_gt}')

    out_path = args.work_dir / args.output
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_dir','video_name','gt','pred','score'])
        writer.writeheader()
        writer.writerows(recovered_bad)

    print(f'\nRecovered {len(recovered_bad)} misclassified entries -> {out_path}')
    if ambiguous:
        print(f'  ({ambiguous} ambiguous positions — check those rows manually)')

    # Print per-confusion-pair summary
    from collections import Counter
    pairs = Counter((r['gt'], r['pred']) for r in recovered_bad)
    print('\nPer-pair summary (gt -> pred, count):')
    for (gt, pred), n in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f'  {gt:12s} -> {pred:12s}  {n}')


if __name__ == '__main__':
    main()
