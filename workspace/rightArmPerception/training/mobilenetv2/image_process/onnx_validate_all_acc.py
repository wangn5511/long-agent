""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import argparse
import numpy as np
import torch
import onnxruntime
from util.utils import AverageMeter
import time
from datasets import MyDataset, build_transform, read_split_data


parser = argparse.ArgumentParser(description='Pytorch ONNX Validation')
parser.add_argument('--data_root', default='D:/flower_data', type=str,
                    help='path to datasets')
parser.add_argument('--onnx-input', default='./mobilenetv4_conv_large_optim.onnx', type=str, metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-output-opt', default='', type=str, metavar='PATH',
                    help='path to output optimized onnx graph')
parser.add_argument('--profile', action='store_true', default=False,
                    help='Enable profiler output.')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16), as same as the train_batch_size in train_gpu.py')
parser.add_argument('--img-size', default=384, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of datasets')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of datasets')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='use tensorflow mnasnet preporcessing')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    args.input_size = args.img_size

    # print(args.input_size)
    # print(args.img_size)

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt

    session = onnxruntime.InferenceSession(args.onnx_input, sess_options, providers=['CPUExecutionProvider'])

    # data_config = resolve_data_config(None, args)
    val_set, class_indices = build_dataset(args)

    loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False
    )

    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_names = [''] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name
    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    end = time.time()
    for i, (input, target) in enumerate(loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]
        preds = np.argmax(output, axis=1)
        targets_np = target.numpy()
        for t, p in zip(targets_np, preds):
            conf_matrix[t, p] += 1

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output, targets_np)
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s, {ms_avg:.3f} ms/sample) \t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, rate_avg=input.size(0) / batch_time.avg,
                ms_avg=100 * batch_time.avg / input.size(0), top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} ({top1a:.3f}) Prec@5 {top5.avg:.3f} ({top5a:.3f})'.format(
        top1=top1, top1a=100-top1.avg, top5=top5, top5a=100.-top5.avg))
    class_metrics = compute_per_class_metrics(conf_matrix)
    print('\nPer-class metrics:')
    for idx, name in enumerate(class_names):
        print(
            '[{idx:02d}] {name:20s} Acc {acc:.2f}%  Precision {prec:.2f}%  Recall {recall:.2f}%  F1 {f1:.2f}%'.format(
                idx=idx,
                name=name,
                acc=class_metrics['accuracy'][idx] * 100,
                prec=class_metrics['precision'][idx] * 100,
                recall=class_metrics['recall'][idx] * 100,
                f1=class_metrics['f1'][idx] * 100
            )
        )
    print('\nMacro Avg -> Acc {acc:.2f}%  Precision {prec:.2f}%  Recall {recall:.2f}%  F1 {f1:.2f}%'.format(
        acc=class_metrics['accuracy'].mean() * 100,
        prec=class_metrics['precision'].mean() * 100,
        recall=class_metrics['recall'].mean() * 100,
        f1=class_metrics['f1'].mean() * 100
    ))


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


def compute_per_class_metrics(conf_matrix):
    tp = np.diag(conf_matrix).astype(np.float64)
    support = conf_matrix.sum(axis=1).astype(np.float64)
    predicted = conf_matrix.sum(axis=0).astype(np.float64)

    accuracy = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def build_dataset(args):
    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_root)

    valid_transform = build_transform(False, args)

    valid_set = MyDataset(val_image_path, val_image_label, valid_transform)
    return valid_set, class_indices


if __name__ == '__main__':
    main()