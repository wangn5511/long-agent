# STGCN Hand21 Clip Length Experiments

## Configs

- `u36`: `training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u36-30e_custom-keypoint-2d.py`
- `u16`: `training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u16-30e_custom-keypoint-2d.py`
- `u8`: `training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u8-30e_custom-keypoint-2d.py`

## Recommended training commands

```bash
python training/mmaction2/tools/train.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u36-30e_custom-keypoint-2d.py \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u36_exp1
```

```bash
python training/mmaction2/tools/train.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u16-30e_custom-keypoint-2d.py \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u16_exp1
```

```bash
python training/mmaction2/tools/train.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u8-30e_custom-keypoint-2d.py \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u8_exp1
```

## Recommended test commands

Replace `best_acc_top1_epoch_*.pth` with the actual best checkpoint filename.

```bash
python training/mmaction2/tools/test.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u36-30e_custom-keypoint-2d.py \
  training/mmaction2/work_dirs/stgcn_hand21_u36_exp1/best_acc_top1_epoch_*.pth \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u36_exp1
```

```bash
python training/mmaction2/tools/test.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u16-30e_custom-keypoint-2d.py \
  training/mmaction2/work_dirs/stgcn_hand21_u16_exp1/best_acc_top1_epoch_*.pth \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u16_exp1
```

```bash
python training/mmaction2/tools/test.py \
  training/mmaction2/configs/skeleton/stgcn/stgcn_hand21-joint-u8-30e_custom-keypoint-2d.py \
  training/mmaction2/work_dirs/stgcn_hand21_u8_exp1/best_acc_top1_epoch_*.pth \
  --work-dir training/mmaction2/work_dirs/stgcn_hand21_u8_exp1
```

## One-click script

From `training/mmaction2/`:

```bash
bash tools/run_stgcn_hand21_cliplen_experiments.sh exp1
```

This script will:

- train `u36`, `u16`, `u8`
- test each experiment with its best checkpoint
- store each run in a separate `work_dir`
- generate a summary file under `work_dirs/`

## Expected outputs per experiment

Each experiment directory should contain:

- `best_acc_top1_epoch_*.pth`
- latest checkpoint file
- `confusion_matrix_test.npy`
- `confusion_matrix_test_summary.txt`
- `misclassified_test.txt`
- `classified_test.txt`
- training logs and json logs

## Notes

- Current `test_dataloader` still uses `split='val'`. If you have a real test split, update the config before comparing final test-set numbers.
- For fair comparison, keep dataset split, epochs, optimizer, batch size, and random seed strategy consistent across `u36`, `u16`, and `u8`.
